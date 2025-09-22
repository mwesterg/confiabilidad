# reliability_pipeline.py
# pip install pandapower pyyaml pandas numpy networkx matplotlib

import argparse
import itertools as it
import yaml
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1) Build network from YAML
# =========================
def _collect_bus_references(data):
    refs = set()
    for l in data.get("lines", []):
        refs.add(str(l["from"]).strip()); refs.add(str(l["to"]).strip())
    for ld in data.get("loads", []):
        refs.add(str(ld["bus"]).strip())
    return refs

def _bus_name_set(data):
    return {str(b["name"]).strip() for b in data.get("buses", [])}

def validate_yaml_topology(data):
    # normalize and validate basic fields
    for b in data.get("buses", []):
        b["name"] = str(b["name"]).strip()
    for l in data.get("lines", []):
        l["name"] = str(l["name"]).strip()
        l["from"] = str(l["from"]).strip()
        l["to"] = str(l["to"]).strip()
    for ld in data.get("loads", []):
        ld["name"] = str(ld["name"]).strip()
        ld["bus"] = str(ld["bus"]).strip()

    defined = _bus_name_set(data)
    refs = _collect_bus_references(data)
    missing = sorted(refs - defined)
    if missing:
        lines_pointing = []
        for l in data.get("lines", []):
            if l["from"] in missing or l["to"] in missing:
                lines_pointing.append(f"- line '{l['name']}': from={l['from']} to={l['to']}")
        loads_pointing = []
        for ld in data.get("loads", []):
            if ld["bus"] in missing:
                loads_pointing.append(f"- load '{ld['name']}': bus={ld['bus']}")
        raise KeyError(
            "YAML references buses not defined under 'buses:'.\n"
            f"Missing: {missing}\nReferenced in:\n" + "\n".join(lines_pointing + loads_pointing)
        )

def build_network(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    validate_yaml_topology(data)

    net = pp.create_empty_network()

    # Default line type (use real data if available)
    if "std_line_type" in data:
        pp.create_std_type(
            net,
            data["std_line_type"]["params"],
            name=data["std_line_type"]["name"],
            element="line",
        )
        default_line_type = data["std_line_type"]["name"]
    else:
        pp.create_std_type(
            net,
            dict(r_ohm_per_km=0.4, x_ohm_per_km=0.3, c_nf_per_km=200.0, max_i_ka=0.4, type="cs"),
            name="MV_std",
            element="line",
        )
        default_line_type = "MV_std"

    # Buses
    bus = {}
    for b in data["buses"]:
        name = str(b["name"]).strip()
        vn_kv = float(b.get("vn_kv", 12.0))
        bus[name] = pp.create_bus(net, vn_kv=vn_kv, name=name)

    # Slack(s)
    for b in data["buses"]:
        if b.get("slack", False):
            pp.create_ext_grid(net, bus=bus[b["name"]], vm_pu=float(b.get("vm_pu", 1.0)), name=f"Slack {b['name']}")

    # Lines
    for l in data.get("lines", []):
        std_type = l.get("type", default_line_type)
        pp.create_line(
            net,
            bus[l["from"]],
            bus[l["to"]],
            length_km=float(l.get("length_km", 0.1)),
            std_type=std_type,
            name=l["name"],
        )

    # Loads (active power only, Q = 0)
    for ld in data.get("loads", []):
        pp.create_load(
            net,
            bus[ld["bus"]],
            p_mw=float(ld["p_kw"]) / 1000.0,
            q_mvar=0.0,
            name=ld["name"],
        )

    # Optional: customers per load for SAIDI/SAIFI
    load_customers = {lc["load"]: int(lc["customers"]) for lc in data.get("load_customers", [])}

    # Reliability per line
    reliability = data.get("reliability", [])

    # Protections (only for plotting; does not change metrics)
    protections = data.get("protections", [])

    return net, reliability, load_customers, protections


# =====================================
# 2) Topology reachability & mode impact
# =====================================
def reachable_buses_from_slack(net):
    """Return set of buses reachable from any slack (ext_grid)."""
    G = top.create_nxgraph(net, respect_switches=True)
    slack_buses = set(net.ext_grid.bus.values)
    reachable = set()
    for sb in slack_buses:
        if sb in G:
            reachable |= nx.descendants(G, sb) | {sb}
    return reachable

def affected_loads_detail(net):
    """List of (load_name, p_mw, bus_name) that are on unreachable buses."""
    reachable = reachable_buses_from_slack(net)
    idx = net.load.index[~net.load.bus.isin(reachable)]
    out = []
    for i in idx:
        ld = net.load.loc[i]
        out.append((net.load.at[i, "name"], float(ld["p_mw"]), net.bus.loc[ld["bus"], "name"]))
    return out

def mode_impact(net, line_names):
    """Open the given lines, compute unsupplied P (MW) and affected loads, restore lines."""
    idxs = net.line.index[net.line.name.isin(line_names)]
    old = net.line.in_service.loc[idxs].copy()
    net.line.in_service.loc[idxs] = False
    try:
        aff = affected_loads_detail(net)
        p_unsup = sum(p for _, p, _ in aff)
    finally:
        net.line.in_service.loc[idxs] = old
    return p_unsup, aff


# ================================================
# 3) ENS, SAIDI, SAIFI for N-1 / N-2 (topological)
# ================================================
def reliability_dataframe(net, reliability, load_customers, max_order=2,
                          n2_drop_small_prob=True, n2_prob_cut=1e-8):
    rel = {r["line"]: {"lambda": float(r["lambda"]), "mttr": float(r["mttr"])} for r in reliability}
    unknown = [ln for ln in rel.keys() if ln not in set(net.line.name)]
    if unknown:
        raise ValueError(f"Lines in 'reliability' not found in network: {unknown}")

    total_customers = sum(load_customers.values()) if load_customers else 0
    rows = []

    # N-1
    for name, pars in rel.items():
        lam, mttr = pars["lambda"], pars["mttr"]
        exp_hours = lam * mttr                   # expected outage hours/year
        p_unsup_mw, aff = mode_impact(net, [name])
        eens = p_unsup_mw * exp_hours           # MWh/year

        if load_customers:
            cust_aff = sum(load_customers.get(ld_name, 0) for ld_name, _, _ in aff)
        else:
            cust_aff = np.nan

        if total_customers and not np.isnan(cust_aff):
            saidi = (exp_hours * cust_aff) / total_customers
            saifi = (lam * cust_aff) / total_customers
        else:
            saidi = np.nan
            saifi = np.nan

        rows.append(dict(
            order=1, mode=(name,), mode_label=name,
            exp_hours=exp_hours, p_unsup_MW=p_unsup_mw, EENS_MWh=eens,
            SAIDI_h=saidi, SAIFI=saifi, customers_affected=cust_aff,
            affected_loads=[ld for ld, _, _ in aff]
        ))

    # N-2
    if max_order >= 2:
        names = list(rel.keys())
        for a, b in it.combinations(names, 2):
            lam1, mttr1 = rel[a]["lambda"], rel[a]["mttr"]
            lam2, mttr2 = rel[b]["lambda"], rel[b]["mttr"]
            q1 = lam1 * mttr1 / 8760.0
            q2 = lam2 * mttr2 / 8760.0
            exp_hours = q1 * q2 * 8760.0  # expected simultaneous hours/year
            if n2_drop_small_prob and exp_hours < (n2_prob_cut * 8760):
                continue

            p_unsup_mw, aff = mode_impact(net, [a, b])
            eens = p_unsup_mw * exp_hours

            if load_customers:
                cust_aff = sum(load_customers.get(ld_name, 0) for ld_name, _, _ in aff)
            else:
                cust_aff = np.nan

            dur_eq = max(mttr1, mttr2)
            lam_eq = exp_hours / dur_eq if dur_eq > 0 else 0.0

            if total_customers and not np.isnan(cust_aff):
                saidi = (exp_hours * cust_aff) / total_customers
                saifi = (lam_eq * cust_aff) / total_customers
            else:
                saidi = np.nan
                saifi = np.nan

            rows.append(dict(
                order=2, mode=(a, b), mode_label=f"{a} & {b}",
                exp_hours=exp_hours, p_unsup_MW=p_unsup_mw, EENS_MWh=eens,
                SAIDI_h=saidi, SAIFI=saifi, customers_affected=cust_aff,
                affected_loads=[ld for ld, _, _ in aff]
            ))

    df = pd.DataFrame(rows).sort_values(["order", "EENS_MWh"], ascending=[True, False]).reset_index(drop=True)
    summary = dict(
        total_EENS_MWh=df["EENS_MWh"].sum(),
        total_SAIDI_h=df["SAIDI_h"].sum(skipna=True),
        total_SAIFI=df["SAIFI"].sum(skipna=True),
    )
    return df, summary


# =========================================================
# 4) N-1 Affects Table: which line outages affect each load
# =========================================================
def compute_affects_table(net):
    detailed_rows = []
    for line_name in net.line.name.tolist():
        p_unsup_mw, aff = mode_impact(net, [line_name])
        if aff:
            for ld_name, p_mw, bus_name in aff:
                detailed_rows.append({
                    "line": line_name,
                    "affected_load": ld_name,
                    "load_bus": bus_name,
                    "load_p_mw": p_mw
                })
    if not detailed_rows:
        return (pd.DataFrame(columns=["line", "affected_load", "load_bus", "load_p_mw"]),
                pd.DataFrame(columns=["load", "affected_by_lines_count", "lines_csv"]))

    df_detail = pd.DataFrame(detailed_rows).sort_values(["affected_load", "line"]).reset_index(drop=True)
    per_load = (df_detail.groupby("affected_load")["line"]
                .agg(lambda s: sorted(set(s)))
                .reset_index(name="lines"))
    per_load["affected_by_lines_count"] = per_load["lines"].apply(len)
    per_load["lines_csv"] = per_load["lines"].apply(lambda L: ", ".join(L))
    per_load = per_load.drop(columns=["lines"]).rename(columns={"affected_load": "load"})
    per_load = per_load.sort_values(["affected_by_lines_count", "load"], ascending=[False, True]).reset_index(drop=True)
    return df_detail, per_load


# ==========================
# 5) Topological plotting (NetworkX only) + protection markers
# ==========================
def _compute_depths(G, slacks):
    UG = G.to_undirected()
    depths = {n: np.inf for n in G.nodes}
    for s in slacks:
        if s in UG:
            lengths = nx.single_source_shortest_path_length(UG, s)
            for n, d in lengths.items():
                depths[n] = min(depths[n], d)
    finite = [d for d in depths.values() if np.isfinite(d)]
    maxd = int(max(finite)) if finite else 0
    for n, d in depths.items():
        if not np.isfinite(d):
            depths[n] = maxd + 1
    return {n: int(d) for n, d in depths.items()}

def _interpolate_point(p0, p1, alpha=0.15):
    """Point at p = (1-alpha)*p0 + alpha*p1."""
    return ( (1-alpha)*p0[0] + alpha*p1[0], (1-alpha)*p0[1] + alpha*p1[1] )

def save_topology_plot(net, protections, png_path="grid_topology.png", dpi=220):
    """
    Draw buses/lines using layered layout and overlay protection markers:
      - CB: square marker
      - RC: circle marker
      - FUSE: triangle marker
    Position is near the specified 'at_bus' end of each line.
    """
    G = top.create_nxgraph(net, respect_switches=True)
    if len(G) == 0:
        raise RuntimeError("Empty graph; check your network definition.")

    slack_buses = set(net.ext_grid.bus.values)
    depths = _compute_depths(G, slack_buses)
    nx.set_node_attributes(G, depths, "layer")
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical", scale=1.0)

    # Draw base grid
    node_colors = ["#1f77b4" if n in slack_buses else "#2ca02c" for n in G.nodes]
    plt.figure(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_edges(G, pos, arrows=False)

    # Bus labels
    labels = {i: net.bus.at[i, "name"] for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # Edge labels with line names
    edge_labels = {}
    for idx, row in net.line.iterrows():
        u, v = int(row.from_bus), int(row.to_bus)
        name = row.name
        if (u, v) in G.edges:
            edge_labels[(u, v)] = name
        elif (v, u) in G.edges:
            edge_labels[(v, u)] = name
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False)

    # Protection markers
    # Map types to matplotlib markers
    marker_map = {"CB": "s", "RC": "o", "FUSE": "^"}
    color_map  = {"CB": "#d62728", "RC": "#9467bd", "FUSE": "#ff7f0e"}  # red, purple, orange

    handles_for_legend = {}
    for p in protections:
        ptype = str(p["type"]).upper()
        line_name = str(p["line"]).strip()
        at_bus_name = str(p["at_bus"]).strip()

        if ptype not in marker_map:
            continue  # unknown type: skip

        # Find line endpoints and bus index for at_bus
        if line_name not in set(net.line.name):
            continue
        line_row = net.line.loc[net.line.name == line_name].iloc[0]
        u, v = int(line_row.from_bus), int(line_row.to_bus)

        bus_idx_series = net.bus.index[net.bus.name == at_bus_name]
        if len(bus_idx_series) == 0:
            continue
        at_bus_idx = int(bus_idx_series[0])

        if at_bus_idx == u:
            other = v
        elif at_bus_idx == v:
            other = u
        else:
            # If at_bus isn't one end, place marker at the closer end
            du = np.linalg.norm(np.array(pos[at_bus_idx]) - np.array(pos[u])) if at_bus_idx in pos else np.inf
            dv = np.linalg.norm(np.array(pos[at_bus_idx]) - np.array(pos[v])) if at_bus_idx in pos else np.inf
            at_bus_idx = u if du <= dv else v
            other = v if at_bus_idx == u else u

        if at_bus_idx not in pos or other not in pos:
            continue

        # Position slightly away from the bus towards the line
        xy = _interpolate_point(pos[at_bus_idx], pos[other], alpha=0.15)
        plt.scatter([xy[0]], [xy[1]], s=160, marker=marker_map[ptype],
                    facecolor=color_map[ptype], edgecolor="black", linewidths=0.8, zorder=5)
        # tiny text label
        plt.text(xy[0], xy[1]+0.02, ptype, ha="center", va="bottom", fontsize=8)

        if ptype not in handles_for_legend:
            handles_for_legend[ptype] = Line2D([0], [0], marker=marker_map[ptype], color="w",
                                               markerfacecolor=color_map[ptype], markeredgecolor="black",
                                               markersize=8, label=ptype)

    if handles_for_legend:
        plt.legend(handles=list(handles_for_legend.values()), loc="upper left", frameon=False)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=dpi)
    plt.close()
    return png_path


# ====================
# 6) CLI entry point
# ====================
def main():
    parser = argparse.ArgumentParser(description="Reliability pipeline (ENS, SAIDI, SAIFI, modes, NX topology plot w/ protections)")
    parser.add_argument("--yaml", required=True, help="Path to YAML network file")
    parser.add_argument("--max_order", type=int, default=2, help="Max mode order (1=N-1, 2=include N-2)")
    parser.add_argument("--top", type=int, default=15, help="How many modes to print")
    parser.add_argument("--plot", action="store_true", help="Save a NetworkX topology PNG (grid_topology.png)")
    args = parser.parse_args()

    net, reliability, load_customers, protections = build_network(args.yaml)

    # Optional: run PF (not required for topology metrics)
    try:
        pp.runpp(net, init="auto")
    except Exception:
        print("Note: base power flow did not converge; continuing with topology-based metrics.")

    # Reliability analysis
    df_modes, summary = reliability_dataframe(net, reliability, load_customers, max_order=args.max_order)

    print("\n=== Top modes by expected EENS ===")
    cols = ["order", "mode_label", "p_unsup_MW", "exp_hours", "EENS_MWh", "SAIDI_h", "SAIFI", "affected_loads"]
    print(df_modes.sort_values("EENS_MWh", ascending=False).head(args.top)[cols])

    print("\n=== Annual expected summary ===")
    print(summary)

    # Save modes table
    df_modes.to_csv("modes_ranking.csv", index=False)
    print("\nSaved: modes_ranking.csv")

    # Affects tables (N-1)
    df_aff_detail, df_aff_per_load = compute_affects_table(net)
    df_aff_detail.to_csv("affected_by_line_detail.csv", index=False)
    df_aff_per_load.to_csv("affected_by_line_per_load.csv", index=False)
    print("Saved: affected_by_line_detail.csv (line → affected loads)")
    print("Saved: affected_by_line_per_load.csv (load → lines that affect it)")

    # Plot (with protection markers if provided)
    if args.plot:
        out_png = save_topology_plot(net, protections, "grid_topology.png", dpi=220)
        print(f"Saved topology plot: {out_png}")

if __name__ == "__main__":
    main()
