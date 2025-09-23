#!/usr/bin/env python3
# main.py
# pip install pandapower pyyaml pandas numpy networkx matplotlib

import argparse
import itertools as it
import yaml
import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# YAML + Network building
# =========================
def validate_yaml_topology(data):
    def norm(s): return str(s).strip()
    bus_names = [norm(b["name"]) for b in data.get("buses", [])]
    dups = {b for b in bus_names if bus_names.count(b) > 1}
    if dups:
        raise ValueError(f"Duplicate bus names: {sorted(dups)}")
    refs = set()
    for l in data.get("lines", []):
        for k in ("name", "from", "to"):
            if k not in l:
                raise ValueError(f"Line missing field '{k}': {l}")
        refs.add(norm(l["from"])); refs.add(norm(l["to"]))
    for ld in data.get("loads", []):
        for k in ("name", "bus"):
            if k not in ld:
                raise ValueError(f"Load missing field '{k}': {ld}")
        refs.add(norm(ld["bus"]))
    missing = sorted(refs - set(bus_names))
    if missing:
        raise KeyError(f"Missing buses in YAML: {missing}")

def build_network(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate_yaml_topology(data)

    net = pp.create_empty_network()

    # Buses
    bus_map = {}
    for b in data.get("buses", []):
        name = str(b["name"]).strip()
        idx = pp.create_bus(net, vn_kv=float(b.get("vn_kv", 12.0)), name=name)
        bus_map[name] = idx

    # Lines (use parameters so we don't rely on std types)
    for l in data.get("lines", []):
        pp.create_line_from_parameters(
            net,
            from_bus=bus_map[str(l["from"]).strip()],
            to_bus=bus_map[str(l["to"]).strip()],
            length_km=float(l.get("length_km", 0.5)),
            r_ohm_per_km=float(l.get("r_ohm_per_km", 0.1)),
            x_ohm_per_km=float(l.get("x_ohm_per_km", 0.1)),
            c_nf_per_km=float(l.get("c_nf_per_km", 0.0)),
            max_i_ka=float(l.get("max_i_ka", 1.0)),
            name=str(l["name"]).strip()
        )

    # Loads (active power only)
    for ld in data.get("loads", []):
        pp.create_load(
            net,
            bus=bus_map[str(ld["bus"]).strip()],
            p_mw=float(ld.get("p_kw", 0.0)) / 1000.0,
            q_mvar=0.0,
            name=str(ld["name"]).strip()
        )

    # Slacks: per-bus flag and/or legacy top-level
    for b in data.get("buses", []):
        if b.get("slack", False):
            pp.create_ext_grid(net, bus=bus_map[str(b["name"]).strip()], vm_pu=1.0, name=f"Slack {b['name']}")
    if "slack" in data and net.ext_grid.empty:
        sb = str(data["slack"]["bus"]).strip()
        if sb in bus_map:
            pp.create_ext_grid(net, bus=bus_map[sb], vm_pu=1.0, name=f"Slack {sb}")

    protections = data.get("protections", [])

    # Reliability section (preferred)
    if "reliability" in data and data["reliability"]:
        reliability = {str(r["line"]).strip(): (float(r.get("lambda", 0.1)), float(r.get("mttr", 5.0)))
                       for r in data["reliability"]}
    else:
        reliability = {str(l["name"]).strip(): (float(l.get("lambda", 0.1)), float(l.get("mttr", 5.0)))
                       for l in data.get("lines", [])}

    # Customers per load (optional; default 1 each)
    load_customers = {}
    for ld in data.get("loads", []):
        load_customers[str(ld["name"]).strip()] = int(ld.get("clients", 1))

    return net, reliability, load_customers, protections

# =========================
# Protection maps (cut-points)
# =========================
def build_cutpoint_map(protections, net):
    """
    Returns: cutpoint_lines_by_bus = {bus_idx: set(line_idx)}
    Only CB and FUSE act as cut-points. RC is ignored for isolation (still plotted).
    """
    name_to_line_idx = {net.line.at[i, "name"]: int(i) for i in net.line.index}
    cutpoint_lines_by_bus = {}
    for p in protections:
        ptype = str(p.get("type", "")).upper()
        if ptype not in {"CB", "FUSE"}:
            continue
        at_bus_name = str(p.get("at_bus", "")).strip()
        line_name = str(p.get("line", "")).strip()
        if not at_bus_name or not line_name or line_name not in name_to_line_idx:
            continue
        bus_idx_series = net.bus.index[net.bus.name == at_bus_name]
        if len(bus_idx_series) == 0:
            continue
        bus_idx = int(bus_idx_series[0])
        line_idx = name_to_line_idx[line_name]
        cutpoint_lines_by_bus.setdefault(bus_idx, set()).add(line_idx)
    return cutpoint_lines_by_bus

# =========================
# Static topology graph (for paths to slack)
# =========================
def static_topology_graph(net):
    """Undirected graph of the full topology (all lines, regardless of in_service)."""
    G = nx.Graph()
    for b in net.bus.index:
        G.add_node(int(b))
    for i, ln in net.line.iterrows():
        G.add_edge(int(ln.from_bus), int(ln.to_bus), line_idx=int(i))
    return G

def nearest_slack_path_nodes(net, start_bus, UG_static):
    """Shortest path (by hops) from start_bus to the nearest slack in the static UG."""
    slacks = set(net.ext_grid.bus.values)
    if start_bus in slacks:
        return [start_bus]
    try:
        lengths = nx.single_source_shortest_path_length(UG_static, start_bus)
        cand = [(s, lengths[s]) for s in slacks if s in lengths]
        if not cand:
            return None
        s_best = min(cand, key=lambda t: t[1])[0]
        return nx.shortest_path(UG_static, start_bus, s_best)
    except Exception:
        return None

# =========================
# Isolation logic (selective, minimal)
# =========================
def extra_lines_to_open_for_faults(net, fault_line_names, protections):
    """
    Minimal selective opening that works correctly when a fault end is itself at a CB bus:
      • Always open the faulted line(s).
      • For each fault end, if the end bus is a cut-point (CB/FUSE), open its declared line(s) and stop on that side.
      • Otherwise walk toward the nearest slack in the STATIC (topology-only) graph and
        open the FIRST cut-point (checking both the current node and the next node at each step).
      • RC devices are ignored for isolation (still plotted).
    """
    import networkx as nx

    # Map: line name -> index
    name2idx = {net.line.at[i, "name"]: int(i) for i in net.line.index}

    # Build cut-point map: {bus_idx: set(line_idx)} for CB/FUSE only
    cutpoints = {}
    for p in protections:
        typ = str(p.get("type", "")).upper()
        if typ not in {"CB", "FUSE"}:
            continue
        bus_name = str(p.get("at_bus", "")).strip()
        line_name = str(p.get("line", "")).strip()
        if not bus_name or not line_name or line_name not in name2idx:
            continue
        bus_idx_series = net.bus.index[net.bus.name == bus_name]
        if len(bus_idx_series) == 0:
            continue
        bus_idx = int(bus_idx_series[0])
        line_idx = name2idx[line_name]
        cutpoints.setdefault(bus_idx, set()).add(line_idx)

    # STATIC undirected graph (topology only) to find shortest path to a slack
    UG = nx.Graph()
    for b in net.bus.index:
        UG.add_node(int(b))
    for i, row in net.line.iterrows():
        UG.add_edge(int(row.from_bus), int(row.to_bus), line_idx=int(i))

    slacks = set(net.ext_grid.bus.values)

    def nearest_slack_path_nodes(start):
        if start in slacks:
            return [start]
        try:
            dists = nx.single_source_shortest_path_length(UG, start)
            candidates = [(s, dists[s]) for s in slacks if s in dists]
            if not candidates:
                return None
            s_best = min(candidates, key=lambda t: t[1])[0]
            return nx.shortest_path(UG, start, s_best)
        except Exception:
            return None

    to_open = set()

    # Always open the faulted line(s)
    for ln in fault_line_names:
        if ln in name2idx:
            to_open.add(name2idx[ln])

    # For each fault end, open ONLY the first cut-point (including if the end is itself a cut-point)
    for ln in fault_line_names:
        if ln not in name2idx:
            continue
        li = name2idx[ln]
        u = int(net.line.at[li, "from_bus"]); v = int(net.line.at[li, "to_bus"])
        for end in (u, v):
            path = nearest_slack_path_nodes(end)
            if not path or len(path) == 0:
                continue

            # Case A: the fault end bus itself is a cut-point
            if end in cutpoints:
                to_open.update(cutpoints[end])
                continue  # stop walking on this side

            # Case B: walk toward slack, stopping at the first cut-point seen
            # Check BOTH the current node (a) and the next node (b)
            for a, b in zip(path[:-1], path[1:]):
                if a in cutpoints:
                    to_open.update(cutpoints[a])
                    break
                if b in cutpoints:
                    to_open.update(cutpoints[b])
                    break
            # If no cut-point found along the way, nothing else to open on this side

    return list(to_open)


# =========================
# Affected loads (UNDIRECTED reachability on current state)
# =========================
def affected_loads_detail(net):
    """
    Returns [(load_name, p_mw, bus_idx)] for loads de-energized under current in_service flags.
    Uses a custom UNDIRECTED graph built only from in-service lines.
    """
    UG = nx.Graph()
    for b in net.bus.index:
        UG.add_node(int(b))
    for i, ln in net.line.iterrows():
        if bool(ln.in_service):
            UG.add_edge(int(ln.from_bus), int(ln.to_bus))

    slacks = set(net.ext_grid.bus.values)
    reachable = set()
    for sb in slacks:
        if sb in UG:
            reachable |= set(nx.single_source_shortest_path_length(UG, int(sb)).keys())

    affected = []
    for _, ld in net.load.iterrows():
        if int(ld.bus) not in reachable:
            affected.append((str(ld["name"]), float(ld.p_mw), int(ld.bus)))
    return affected

# =========================
# Mode impact (returns p_unsup, affected list)
# =========================
def mode_impact(net, fault_line_names, protections):
    """
    Open the faulted line(s) + minimal cut-point CB/FUSE(s), compute impact, then restore.
    Returns:
      p_unsup (MW), affected [(load_name, p_mw, bus_idx)]
    """
    name2idx = {net.line.at[i, "name"]: int(i) for i in net.line.index}
    extra = extra_lines_to_open_for_faults(net, fault_line_names, protections)

    to_open_idxs = set()
    for ln in fault_line_names:
        if ln in name2idx:
            to_open_idxs.add(name2idx[ln])
    to_open_idxs |= set(extra)

    idxs = list(to_open_idxs)
    old = net.line.in_service.loc[idxs].copy()
    net.line.in_service.loc[idxs] = False
    try:
        aff = affected_loads_detail(net)
        p_unsup = sum(p for _, p, _ in aff)
    finally:
        net.line.in_service.loc[idxs] = old
    return p_unsup, aff

# =========================
# Reliability analysis
# =========================
def reliability_analysis(net, reliability, load_customers, protections, max_order=1):
    """
    Returns a DataFrame with per-mode metrics and metadata:
      - EENS_MWh, SAIFI, SAIDI
      - affected_loads_list (list[str]), affected_loads_csv
      - lambda_per_yr, mttr_h
    """
    rows = []

    def summarize(aff):
        names = [str(ld) for ld, _, _ in aff]
        buses = [int(b) for _, _, b in aff]
        p_sum = sum(float(p) for _, p, _ in aff)
        return {
            "affected_loads_list": names,
            "affected_loads_csv": ", ".join(names),
            "affected_loads_count": len(names),
            "affected_buses_csv": ", ".join(str(b) for b in buses),
            "p_unsup_MW": p_sum
        }

    # ---------- N-1 ----------
    for _, row in net.line.iterrows():
        name = row.name if isinstance(row.name, str) else row["name"]
        lam, mttr = reliability.get(name, (0.1, 5.0))   # lam: 1/yr, mttr: hours
        p_unsup, aff = mode_impact(net, [name], protections)
        s = summarize(aff)

        EENS = p_unsup * lam * mttr                      # MW * h/yr = MWh/yr
        SAIFI = sum(load_customers.get(ld, 1) for ld, _, _ in aff) * lam
        SAIDI = sum(load_customers.get(ld, 1) for ld, _, _ in aff) * lam * mttr

        total_clients = sum(load_customers.values())
        SAIFI_norm = SAIFI / total_clients if total_clients > 0 else 0
        SAIDI_norm = SAIDI / total_clients if total_clients > 0 else 0

        rows.append({
            "order": 1,
            "mode": f"N-1:{name}",
            "lines": [name],
            "lambda_per_yr": lam,
            "mttr_h": mttr,
            "EENS_MWh": EENS,
            "SAIFI": SAIFI,
            "SAIDI": SAIDI,
            "SAIFI_norm": SAIFI_norm,
            "SAIDI_norm": SAIDI_norm,
            "p_unsup_MW": s["p_unsup_MW"],
            "affected_loads_count": s["affected_loads_count"],
            "affected_loads_csv": s["affected_loads_csv"],
            "affected_loads_list": s["affected_loads_list"],
            "affected_buses_csv": s["affected_buses_csv"],
        })

    # ---------- N-2 (optional) ----------
    if max_order >= 2:
        names = [row.name for _, row in net.line.iterrows()]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                lam1, mttr1 = reliability.get(a, (0.1, 5.0))
                lam2, mttr2 = reliability.get(b, (0.1, 5.0))
                lam = lam1 * lam2                     # simple approximation
                mttr = (mttr1 + mttr2) / 2.0          # simple approximation

                p_unsup, aff = mode_impact(net, [a, b], protections)
                s = summarize(aff)

                EENS = p_unsup * lam * mttr
                SAIFI = sum(load_customers.get(ld, 1) for ld, _, _ in aff) * lam
                SAIDI = sum(load_customers.get(ld, 1) for ld, _, _ in aff) * lam * mttr

                total_clients = sum(load_customers.values())
                SAIFI_norm = SAIFI / total_clients if total_clients > 0 else 0
                SAIDI_norm = SAIDI / total_clients if total_clients > 0 else 0

                rows.append({
                    "order": 2,
                    "mode": f"N-2:{a}+{b}",
                    "lines": [a, b],
                    "lambda_per_yr": lam,
                    "mttr_h": mttr,
                    "EENS_MWh": EENS,
                    "SAIFI": SAIFI,
                    "SAIDI": SAIDI,
                    "SAIFI_norm": SAIFI_norm,
                    "SAIDI_norm": SAIDI_norm,
                    "p_unsup_MW": s["p_unsup_MW"],
                    "affected_loads_count": s["affected_loads_count"],
                    "affected_loads_csv": s["affected_loads_csv"],
                    "affected_loads_list": s["affected_loads_list"],
                    "affected_buses_csv": s["affected_buses_csv"],
                })

    df = pd.DataFrame(rows).sort_values(["order", "EENS_MWh"], ascending=[True, False]).reset_index(drop=True)
    return df


# =========================
# Per-load “affected by” tables (N-1)
# =========================
def compute_affects_table(net, protections):
    detailed_rows = []
    for line_name in net.line.name.tolist():
        _, aff = mode_impact(net, [line_name], protections)
        for ld_name, p_mw, bus_idx in aff:
            detailed_rows.append({
                "line": str(line_name),
                "affected_load": str(ld_name),
                "load_bus": int(bus_idx),
                "load_p_mw": float(p_mw)
            })

    if not detailed_rows:
        return (pd.DataFrame(columns=["line", "affected_load", "load_bus", "load_p_mw"]),
                pd.DataFrame(columns=["load", "affected_by_lines_count", "lines_csv"]))

    df_detail = pd.DataFrame(detailed_rows).sort_values(["affected_load", "line"]).reset_index(drop=True)
    per_load = (df_detail.groupby("affected_load")["line"]
                .agg(lambda s: sorted(set(s))).reset_index(name="lines"))
    per_load["affected_by_lines_count"] = per_load["lines"].apply(len)
    per_load["lines_csv"] = per_load["lines"].apply(lambda L: ", ".join(L))
    per_load = per_load.drop(columns=["lines"]).rename(columns={"affected_load": "load"})
    per_load = per_load.sort_values(["affected_by_lines_count", "load"], ascending=[False, True]).reset_index(drop=True)
    return df_detail, per_load

# =========================
# Plotting (same style you liked)
# =========================
def _compute_layers_for_layout(net):
    UG = static_topology_graph(net)
    slacks = set(net.ext_grid.bus.values)
    layers = {n: 0 for n in UG.nodes}
    # min hop distance to any slack
    for n in UG.nodes:
        try:
            d = min(nx.shortest_path_length(UG, s, n) for s in slacks if s in UG)
        except ValueError:
            d = 0
        layers[n] = d
    return layers

def _interpolate_point(p0, p1, alpha=0.15):
    return ((1 - alpha) * p0[0] + alpha * p1[0], (1 - alpha) * p0[1] + alpha * p1[1])

def save_topology_plot(net, protections, png_path="topology.png", dpi=200):
    UG = static_topology_graph(net)
    layers = _compute_layers_for_layout(net)
    nx.set_node_attributes(UG, layers, "layer")
    pos = nx.multipartite_layout(UG, subset_key="layer", align="vertical", scale=1.0)

    plt.figure(figsize=(9, 6))
    # edges
    nx.draw_networkx_edges(UG, pos, edge_color="gray", width=1.2)
    # nodes
    slacks = set(net.ext_grid.bus.values)
    node_colors = ["#1f77b4" if n in slacks else "#2ca02c" for n in UG.nodes]
    nx.draw_networkx_nodes(UG, pos, nodelist=list(net.bus.index), node_color=node_colors,
                           node_shape="o", node_size=700, edgecolors="black", linewidths=0.7)
    nx.draw_networkx_labels(UG, pos, labels={i: net.bus.at[i, "name"] for i in net.bus.index}, font_size=8)

    # line labels
    edge_labels = {}
    for i, row in net.line.iterrows():
        u, v = int(row.from_bus), int(row.to_bus)
        if (u, v) in UG.edges:
            edge_labels[(u, v)] = str(row.name)
        elif (v, u) in UG.edges:
            edge_labels[(v, u)] = str(row.name)
    if edge_labels:
        nx.draw_networkx_edge_labels(UG, pos, edge_labels=edge_labels, font_size=7, rotate=False)

    # loads as green triangles with names (stacked a bit below the bus)
    loads_per_bus = {}
    for _, ld in net.load.iterrows():
        b = int(ld.bus)
        loads_per_bus.setdefault(b, []).append(str(ld["name"]))
    for b, names in loads_per_bus.items():
        if b not in pos: continue
        x, y = pos[b]
        for k, nm in enumerate(names):
            y_off = y - 0.10 - 0.06 * k
            plt.scatter([x], [y_off], marker="^", s=420, color="lightgreen",
                        edgecolor="black", linewidths=0.7, zorder=4)
            plt.text(x, y_off - 0.03, nm, fontsize=7, ha="center", va="top")

    # protection markers near the 'at_bus' end on the line
    marker_map = {"CB": "s", "RC": "o", "FUSE": "^"}
    color_map = {"CB": "#d62728", "RC": "#9467bd", "FUSE": "#ff7f0e"}
    legend_handles = {}

    for p in protections:
        typ = str(p.get("type", "")).upper()
        line_name = str(p.get("line", "")).strip()
        at_bus_name = str(p.get("at_bus", "")).strip()
        if typ not in marker_map or not line_name or not at_bus_name:
            continue

        # find line endpoints and target bus
        if line_name not in set(net.line.name):
            continue
        line_row = net.line.loc[net.line.name == line_name].iloc[0]
        u, v = int(line_row.from_bus), int(line_row.to_bus)

        bus_idx_series = net.bus.index[net.bus.name == at_bus_name]
        if len(bus_idx_series) == 0:
            continue
        at_bus_idx = int(bus_idx_series[0])
        other = v if at_bus_idx == u else u if at_bus_idx == v else v

        if at_bus_idx not in pos or other not in pos:
            continue
        xy = _interpolate_point(pos[at_bus_idx], pos[other], alpha=0.15)
        plt.scatter([xy[0]], [xy[1]], s=160, marker=marker_map[typ],
                    facecolor=color_map[typ], edgecolor="black", linewidths=0.8, zorder=5)
        plt.text(xy[0], xy[1] + 0.02, typ, ha="center", va="bottom", fontsize=8)
        if typ not in legend_handles:
            legend_handles[typ] = Line2D([0], [0], marker=marker_map[typ], color="w",
                                         markerfacecolor=color_map[typ], markeredgecolor="black",
                                         markersize=8, label=typ)

    if legend_handles:
        plt.legend(handles=list(legend_handles.values()), loc="upper left", frameon=False)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=dpi)
    plt.close()
    return png_path


def per_load_indices(results_df, net, load_customers):
    """
    Compute ENS (MWh/yr), SAIFI (int/yr), SAIDI (h/yr) per load.
    results_df must have: affected_loads_list, lambda_per_yr, mttr_h.
    """
    load_names = list(net.load["name"])
    # map load -> P (MW)
    load_p_mw = {str(r["name"]): float(r["p_mw"]) for _, r in net.load.iterrows()}

    out = []
    total_clients = sum(load_customers.values())

    for ld in load_names:
        ens = 0.0
        saifi = 0.0
        saidi = 0.0
        for _, r in results_df.iterrows():
            if ld in r["affected_loads_list"]:
                lam = float(r["lambda_per_yr"])
                mttr = float(r["mttr_h"])
                ens  += load_p_mw[ld] * lam * mttr      # MWh/yr
                saifi += lam                             # int/yr
                saidi += lam * mttr                      # h/yr
        
        saifi_norm = saifi / total_clients if total_clients > 0 else 0
        saidi_norm = saidi / total_clients if total_clients > 0 else 0

        out.append({"load": ld, "ENS_MWh": ens, "SAIFI": saifi, "SAIDI_h": saidi, "SAIFI_norm": saifi_norm, "SAIDI_norm": saidi_norm})
    return pd.DataFrame(out)



# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Protection-aware reliability + topology with loads & protections")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--max_order", type=int, default=1, help="1=N-1; 2=add N-2")
    ap.add_argument("--plot", action="store_true", help="Save topology.png")
    args = ap.parse_args()

    net, reliability, load_customers, protections = build_network(args.yaml)

    # reliability
    df = reliability_analysis(net, reliability, load_customers, protections, max_order=args.max_order)
    df.to_csv("modes_ranking.csv", index=False)
    print("Saved: modes_ranking.csv")

    # Compute per-load reliability indices
    per_load = per_load_indices(df, net, load_customers)
    per_load.to_csv("per_load_indices.csv", index=False)

    print("\n=== Per-load reliability indices ===")
    print(per_load.to_string(index=False))

    # per-load effects (N-1)
    det, per = compute_affects_table(net, protections)
    det.to_csv("affected_by_line_detail.csv", index=False)
    per.to_csv("affected_by_line_per_load.csv", index=False)
    print("Saved: affected_by_line_detail.csv, affected_by_line_per_load.csv")

    if args.plot:
        out = save_topology_plot(net, protections, "topology.png", dpi=200)
        print(f"Saved: {out}")

    print("Done.")

if __name__ == "__main__":
    main()
