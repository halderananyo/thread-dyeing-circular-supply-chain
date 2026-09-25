# ═══════════════════════════════════════════════════════════════════════════
# CELL 13 — SENSITIVITY & SCENARIO ANALYSIS (Prediction-Uncertainty Robustness)
# ═══════════════════════════════════════════════════════════════════════════
# Purpose: Address the "deterministic MILP fed by uncertain ML" critique.
# Method : Propagate XGBoost prediction error into the optimizer by solving the
#          SAME MILP under three demand scenarios per day:
#             1. Baseline    : XGBoost point prediction
#             2. Pessimistic : prediction + 1.96 * RMSE  (≈ upper 95% bound, high-waste)
#             3. Optimistic  : prediction - 1.96 * RMSE  (≈ lower 95% bound, low-waste)
#          RMSE is computed PER WASTE STREAM from the held-out actual vs predicted
#          values, so the bounds are empirical (not assumed).
# Output : Stability table + the headline robustness metric:
#          "% of test days on which the optimal set of activated CCs (y_j) is
#           IDENTICAL across all three scenarios."
# Note   : Only data["wastes"] changes between scenarios; every cost/CC/ETP
#          parameter is copied unchanged from the Cell 5 `data` dict.
# ═══════════════════════════════════════════════════════════════════════════
import copy
import numpy as np
import pandas as pd

# ── 1. Empirical per-stream RMSE from the prediction file ────────────────────
_err_df = pd.read_csv(PREDICTIONS_CSV)

_stream_cols = {
    "solid":    ("Solid_Waste_kg_Actual",    "Solid_Waste_kg_Predicted"),
    "yarn":     ("Yarn_Waste_kg_Actual",     "Yarn_Waste_kg_Predicted"),
    "chemical": ("Chemical_Waste_kg_Actual", "Chemical_Waste_kg_Predicted"),
}

RMSE = {}
print("Empirical prediction error (XGBoost, held-out 2024):")
print(f"  {'Stream':10s} {'RMSE':>10s} {'1.96*RMSE':>12s}")
for w, (acol, pcol) in _stream_cols.items():
    a = _err_df[acol].to_numpy(dtype=float)
    p = _err_df[pcol].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    RMSE[w] = rmse
    print(f"  {w:10s} {rmse:10.2f} {1.96*rmse:12.2f}")
print()

# ── 2. Scenario definitions ──────────────────────────────────────────────────
# delta multiplier applied to (1.96 * RMSE) for each stream
SCENARIOS = {
    "Baseline":     0.0,
    "Pessimistic": +1.0,   # prediction + 1.96*RMSE  (more waste than expected)
    "Optimistic":  -1.0,   # prediction - 1.96*RMSE  (less waste than expected)
}

def _perturbed_wastes(base_wastes, direction):
    """Return a new wastes dict shifted by direction * 1.96 * RMSE per stream.
    Quantities are floored at 0 (negative waste is physically meaningless)."""
    out = {}
    for w, q in base_wastes.items():
        shift = direction * 1.96 * RMSE.get(w, 0.0)
        out[w] = max(0.0, float(q) + shift)
    return out

# ── 3. Build the test-date list — FULL prediction range ──────────────────────
#     Uses every date present in the predictions DataFrame `df` (2024 test set).
#     To restrict to a shorter window instead, replace the line below with:
#         _dates = pd.date_range("2024-01-08", "2024-01-14", freq="D")
_dates = pd.to_datetime(df.index)

# `df` is the prediction DataFrame indexed by Date (built in Cell 3).
def _wastes_for_date(date):
    key = pd.Timestamp(date)
    if key not in df.index:
        return None
    row = df.loc[key]
    return {
        "solid":    float(row["Solid_Waste_kg_Predicted"]),
        "yarn":     float(row["Yarn_Waste_kg_Predicted"]),
        "chemical": float(row["Chemical_Waste_kg_Predicted"]),
    }

# ── 4. Solve the MILP for every (date, scenario) ──────────────────────────────
rows = []
_valid_dates = [d for d in _dates if _wastes_for_date(d) is not None]
print(f"Running scenario analysis over {len(_valid_dates)} dates "
      f"x {len(SCENARIOS)} scenarios = {len(_valid_dates)*len(SCENARIOS)} CPLEX solves...")
print("(Progress printed every 30 days; any decision change is always shown.)\n")

for _k, date in enumerate(_valid_dates):
    base_w = _wastes_for_date(date)

    per_date = {"Date": pd.Timestamp(date).strftime("%Y-%m-%d")}
    active_sets = {}
    for sname, direction in SCENARIOS.items():
        d = copy.deepcopy(data)                     # all cost/CC params unchanged
        d["wastes"] = _perturbed_wastes(base_w, direction)
        res = build_and_solve_cplex(d)
        # active_ccs from the solver are CC *indices* (J = range(len(ccs))).
        # Map each index to its CC name so the report is human-readable.
        active = tuple(sorted(
            data["ccs"][j]["name"] if isinstance(j, int) else str(j)
            for j in res["active_ccs"]
        ))
        active_sets[sname] = active
        per_date[f"Z_{sname}"]   = round(res["Z"], 2)
        per_date[f"nCC_{sname}"] = len(active)
        per_date[f"CCs_{sname}"] = " | ".join(active) if active else "(none)"

    base_set = active_sets["Baseline"]
    stable = all(active_sets[s] == base_set for s in SCENARIOS)
    per_date["CC_set_stable"] = stable
    rows.append(per_date)

    # Throttled progress: heartbeat every 30 days, plus every decision change.
    if (_k % 30 == 0) or (not stable):
        flag = "stable" if stable else "*** CC SET CHANGED ***"
        print(f"  [{_k+1:>3}/{len(_valid_dates)}] {per_date['Date']}  "
              f"Z[base]={per_date['Z_Baseline']:>12,.0f}  y_j {flag}")

scenario_df = pd.DataFrame(rows)

# ── 5. Headline robustness metric ─────────────────────────────────────────────
n_total  = len(scenario_df)
n_stable = int(scenario_df["CC_set_stable"].sum()) if n_total else 0
pct      = (100.0 * n_stable / n_total) if n_total else 0.0

print("\n" + "=" * 64)
print("ROBUSTNESS SUMMARY")
print("=" * 64)
print(f"  Test days analysed                : {n_total}")
print(f"  Days with IDENTICAL activated CCs : {n_stable}")
print(f"  Network-decision stability        : {pct:.1f}%")
print("  (Activated collection centers y_j unchanged under")
print("   +/-1.96*RMSE worst-case XGBoost prediction error on these days.)")

# Per-CC activation frequency across ALL solves (robust core vs marginal CCs)
print("\n  Per-CC activation frequency (share of days each CC is opened):")
_cc_counts = {}
for s in SCENARIOS:
    for cell in scenario_df[f"CCs_{s}"]:
        if cell == "(none)":
            continue
        for cc in cell.split(" | "):
            _cc_counts.setdefault(cc, {sc: 0 for sc in SCENARIOS})
            _cc_counts[cc][s] += 1
if _cc_counts:
    hdr = f"    {'Collection Center':38s}" + "".join(f"{s[:5]:>9s}" for s in SCENARIOS)
    print(hdr)
    for cc, c in sorted(_cc_counts.items(), key=lambda kv: -sum(kv[1].values())):
        line = f"    {cc:38s}" + "".join(f"{100.0*c[s]/n_total:>8.0f}%" for s in SCENARIOS)
        print(line)

# Z range across scenarios (cost band for figures / error bars)
print("\n  Objective Z across scenarios (BDT):")
for s in SCENARIOS:
    col = scenario_df[f"Z_{s}"]
    print(f"    {s:12s}  mean={col.mean():>12,.0f}   "
          f"min={col.min():>12,.0f}   max={col.max():>12,.0f}")
print("=" * 64)

# ── 6. Save for the manuscript ────────────────────────────────────────────────
_out_csv = f"{OUTPUT_FOLDER}\\scenario_analysis_robustness.csv"
scenario_df.to_csv(_out_csv, index=False)
print(f"\n✅ Saved: {_out_csv}")
scenario_df
