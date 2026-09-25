"""
============================================================================================
REVERSE LOGISTICS OPTIMIZATION — MILP via DOcplex (IBM CPLEX)
============================================================================================
Thesis: Predictive Waste Classification and Reverse Logistics Optimization Using
        Machine Learning in a Circular Supply Chain
Case Study: Thread Dyeing Company

WASTE STREAMS:
  W = {solid, yarn, chemical}  → Sent to Collection Centers (CC)
  Treated Wastewater           → Recycled at ETP in-factory; remainder drained

NETWORK:  Factory  →  CC_j  →  Processing Facility (within CC_j)

============================================================================================
FULL MILP FORMULATION
============================================================================================

SETS & INDICES
  W  = set of waste types  {solid, yarn, chemical}
  J  = set of candidate Collection Centers  {1, 2, ..., n}

DECISION VARIABLES
  x[w,j]  ∈ ℝ⁺   kg of waste type w sent to CC j            (continuous)
  y[j]    ∈ {0,1} 1 if CC j is contracted/activated           (binary)

PARAMETERS (per CC j, per waste type w)
  D[w]        = predicted daily waste quantity (kg)
  CAP[w,j]    = processing capacity of CC j for waste w (kg)
  R[w,j]      = revenue per kg offered by CC j for waste w  (solid, yarn only)
  F[w,j]      = disposal fee per kg charged by CC j for chemical w
  d[j]        = distance from factory to CC j (km)
  tc[w]       = transport cost per kg per km for waste type w
  h[w]        = holding & preparation cost per kg at factory for waste w
  fc[j]       = fixed contract activation cost for CC j (BDT)
  T[w,j]      = processing days for waste w at CC j
  r           = daily working capital rate (opportunity cost rate, e.g. 0.0003/day)
  co2[w,j]    = CO2 emission per kg for waste w at CC j
  circ[w,j]   = circularity rate (0–1) for waste w at CC j
  P_co2       = CO2 penalty per kg CO2-equivalent (BDT)
  B_circ      = circularity bonus per kg processed (BDT)

OBJECTIVE: Maximize Z
  Z = Σ_{w∈{solid,yarn}} Σ_j  R[w,j] · x[w,j]          (revenue from solid & yarn)
    - Σ_j  F[chemical,j] · x[chemical,j]                 (disposal cost for chemical)
    - Σ_w Σ_j  tc[w] · d[j] · x[w,j]                    (transportation cost)
    - Σ_w Σ_j  h[w] · x[w,j]                             (holding & prep cost)
    - Σ_j  fc[j] · y[j]                                  (★ fixed contract cost)
    - Σ_w Σ_j  (r · T[w,j]) · R_effective[w,j] · x[w,j] (★ working capital/time cost)
    - Σ_w Σ_j  P_co2 · co2[w,j] · x[w,j]                (environmental penalty)
    + Σ_w Σ_j  B_circ · circ[w,j] · x[w,j]              (circularity bonus)
    + ETP_net                                             (treated wastewater value)

  where:
    R_effective[w,j] = R[w,j] for solid/yarn  (revenue delayed by T days)
                     = F[w,j] for chemical     (fee payment delayed by T days)
    Working capital cost = r · T[w,j] · R_effective[w,j] · x[w,j]
    (Rationale: money is tied up / delayed for T[w,j] days before transaction completes;
     r is the daily opportunity cost of capital — e.g. bank rate / 365)

  ETP_net = (recycling_rate · water_L · reuse_value_per_L)
           - ((1 - recycling_rate) · water_L · drain_cost_per_L)

CONSTRAINTS
  (1) Full allocation:  Σ_j x[w,j] = D[w]            ∀w ∈ W
  (2) Capacity-link:   x[w,j] ≤ CAP[w,j] · y[j]      ∀w,j
  (3) Non-negativity:  x[w,j] ≥ 0                     ∀w,j
  (4) Binary:          y[j] ∈ {0,1}                   ∀j

============================================================================================
HOW TO RUN
============================================================================================

OPTION A — IBM CPLEX Academic (Recommended for Thesis)
  1. Register at: https://academic.ibm.com/a2mt/
  2. Download IBM ILOG CPLEX Optimization Studio (free for academics)
  3. Install DOcplex:
       pip install docplex
  4. Set CPLEX path (if not auto-detected):
       In your script or environment: export CPLEX_STUDIO_BINARIES=/path/to/cplex/bin/x86-64_linux
  5. Run:
       python reverse_logistics_cplex.py

OPTION B — CPLEX Community Edition (Limited to 1000 vars/constraints)
  pip install cplex docplex
  This model has ~20-30 variables — well within the community limit.
  Run as above.

OPTION C — DOcplex Cloud (No local install)
  pip install docplex
  The script will automatically fall back to DOcplex cloud solver if CPLEX is not installed.
  You need an IBM Cloud account with Decision Optimization service.

OPTION D — Fallback to scipy (for testing without CPLEX)
  If neither cplex nor cloud is available, set USE_SCIPY_FALLBACK = True below.

============================================================================================
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
USE_SCIPY_FALLBACK = False   # Set True if CPLEX not available (for testing)
DAILY_WORKING_CAPITAL_RATE = 0.0003  # 0.03% per day ≈ 10% annual rate / 365

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import sys

def check_docplex():
    try:
        from docplex.mp.model import Model
        return True
    except ImportError:
        return False

def check_cplex():
    try:
        import cplex
        return True
    except ImportError:
        return False

# ── INTERACTIVE INPUT UTILITIES ───────────────────────────────────────────────

def header(text):
    print("\n" + "═"*65)
    print(f"  {text}")
    print("═"*65)

def section(text):
    print(f"\n  ─── {text} ───")

def ask_float(prompt, default, unit="", min_val=None, max_val=None):
    """Ask user for a float value with default."""
    unit_str = f" [{unit}]" if unit else ""
    while True:
        try:
            raw = input(f"    {prompt}{unit_str} (default={default}): ").strip()
            val = float(raw) if raw else default
            if min_val is not None and val < min_val:
                print(f"    ⚠ Value must be ≥ {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"    ⚠ Value must be ≤ {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("    ⚠ Please enter a valid number.")

def ask_int(prompt, default, min_val=1):
    while True:
        try:
            raw = input(f"    {prompt} (default={default}): ").strip()
            val = int(raw) if raw else default
            if val < min_val:
                print(f"    ⚠ Must be ≥ {min_val}.")
                continue
            return val
        except ValueError:
            print("    ⚠ Please enter a valid integer.")

def ask_str(prompt, default):
    raw = input(f"    {prompt} (default='{default}'): ").strip()
    return raw if raw else default

def confirm(prompt):
    raw = input(f"    {prompt} [y/n] (default=y): ").strip().lower()
    return raw != "n"

# ── MAIN INPUT COLLECTION ─────────────────────────────────────────────────────

def collect_inputs():
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   REVERSE LOGISTICS MILP OPTIMIZER — Thread Dyeing Factory  ║")
    print("║   DOcplex (IBM CPLEX) · Interactive Input Mode              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n  Press ENTER to accept default values shown in parentheses.\n")

    # ── 1. Daily Waste Quantities ────────────────────────────────────────────
    header("STEP 1: Daily Predicted Waste Quantities")
    print("  (From your ML model predictions for the target day)\n")
    wastes = {
        "solid":    ask_float("Solid Waste",    1665.87, "kg"),
        "yarn":     ask_float("Yarn Waste",      722.36, "kg"),
        "chemical": ask_float("Chemical Sludge", 3202.32, "kg"),
    }
    treated_ww = ask_float("Treated Wastewater", 451586.56, "litres")

    # ── 2. Factory Costs ─────────────────────────────────────────────────────
    header("STEP 2: Factory Holding & Preparation Costs")
    print("  (Cost per kg to hold and prepare each waste type for dispatch)\n")
    holding = {
        "solid":    ask_float("Solid   — Holding & Prep Cost", 2.5,  "BDT/kg"),
        "yarn":     ask_float("Yarn    — Holding & Prep Cost", 3.0,  "BDT/kg"),
        "chemical": ask_float("Chemical — Holding & Prep Cost", 4.0, "BDT/kg"),
    }

    # ── 3. Transport Costs ───────────────────────────────────────────────────
    header("STEP 3: Transportation Costs (per kg per km)")
    print("  (Varies by waste type due to vehicle requirements / hazmat)\n")
    transport = {
        "solid":    ask_float("Solid   — Transport Cost", 0.05, "BDT/kg/km"),
        "yarn":     ask_float("Yarn    — Transport Cost", 0.04, "BDT/kg/km"),
        "chemical": ask_float("Chemical — Transport Cost", 0.08, "BDT/kg/km (hazmat premium)"),
    }

    # ── 4. ETP Parameters ────────────────────────────────────────────────────
    header("STEP 4: ETP — Treated Wastewater (In-Factory Recycling)")
    print("  (Treated wastewater recycled at ETP; rest drained)\n")
    etp = {
        "recycling_rate":    ask_float("ETP Recycling Rate", 0.75, "0 to 1", 0.0, 1.0),
        "reuse_value_per_L": ask_float("Water Reuse Value",  0.002,  "BDT/litre"),
        "drain_cost_per_L":  ask_float("Drain/Disposal Cost", 0.0005, "BDT/litre"),
    }

    # ── 5. Working Capital Rate ──────────────────────────────────────────────
    header("STEP 5: Working Capital / Time-Value Parameters")
    print("  (Revenue/fee transaction completes only AFTER processing time.)")
    print("  (Capital is tied up during processing → opportunity cost.)\n")
    wc_rate = ask_float(
        "Daily working capital rate (e.g. 0.0003 = 10% annual / 365)",
        DAILY_WORKING_CAPITAL_RATE, "per day"
    )

    # ── 6. Environmental Parameters ──────────────────────────────────────────
    header("STEP 6: Environmental Parameters")
    co2_penalty      = ask_float("CO2 Penalty",         1.5, "BDT per kg CO2-equivalent")
    circularity_bonus = ask_float("Circularity Bonus",  2.0, "BDT per kg × circularity rate")

    # ── 7. Collection Centers ────────────────────────────────────────────────
    header("STEP 7: Collection Centers (CCs)")
    num_ccs = ask_int("How many candidate CCs do you have?", 3)

    waste_types = ["solid", "yarn", "chemical"]
    ccs = []

    for j in range(num_ccs):
        section(f"CC {j+1} of {num_ccs}")
        cc = {}
        cc["name"]     = ask_str(f"  CC {j+1} Name", f"CollectionCenter_{j+1}")
        cc["distance"] = ask_float(f"  Distance from factory", [45,30,20][j] if j < 3 else 40, "km")
        cc["fixed_contract_cost"] = ask_float(
            f"  Fixed contract/activation cost",
            [5000, 4000, 3000][j] if j < 3 else 4000,
            "BDT (paid once if CC is activated)"
        )
        cc["facilities"] = {}

        for w in waste_types:
            section(f"    {w.upper()} Facility at {cc['name']}")
            fac = {}
            fac["capacity"]    = ask_float(f"    Capacity",     [2000,1500,1000][j] if j<3 else 800, "kg/day")
            if w in ["solid", "yarn"]:
                fac["revenue_per_kg"] = ask_float(
                    f"    Revenue per kg offered",
                    {"solid":[8,7,6.5],"yarn":[12,11.5,10]}[w][j] if j<3 else 7.0,
                    "BDT/kg"
                )
                fac["disposal_fee_per_kg"] = 0.0
            else:  # chemical
                fac["disposal_fee_per_kg"] = ask_float(
                    f"    Disposal fee per kg charged",
                    [5.0, 6.0, 4.5][j] if j<3 else 5.0,
                    "BDT/kg"
                )
                fac["revenue_per_kg"] = 0.0
            fac["process_days"]      = ask_int(  f"    Processing time", [7,6,8][j] if j<3 else 7)
            fac["co2_per_kg"]        = ask_float(f"    CO2 emissions",   [0.8,0.9,1.0][j] if j<3 else 0.9, "kg CO2/kg waste")
            fac["circularity_rate"]  = ask_float(
                f"    Circularity rate (0-1)",
                {"solid":[0.85,0.80,0.75],"yarn":[0.90,0.88,0.82],"chemical":[0.20,0.15,0.25]}[w][j] if j<3 else 0.5,
                "0 to 1", 0.0, 1.0
            )
            cc["facilities"][w] = fac

        ccs.append(cc)

    return {
        "wastes": wastes,
        "treated_ww": treated_ww,
        "holding": holding,
        "transport": transport,
        "etp": etp,
        "wc_rate": wc_rate,
        "co2_penalty": co2_penalty,
        "circularity_bonus": circularity_bonus,
        "ccs": ccs,
    }


# ── CPLEX MODEL ───────────────────────────────────────────────────────────────

def build_and_solve_cplex(data):
    from docplex.mp.model import Model

    wastes     = data["wastes"]
    treated_ww = data["treated_ww"]
    holding    = data["holding"]
    transport  = data["transport"]
    etp        = data["etp"]
    wc_rate    = data["wc_rate"]
    co2_pen    = data["co2_penalty"]
    circ_bonus = data["circularity_bonus"]
    ccs        = data["ccs"]

    W = list(wastes.keys())   # ["solid", "yarn", "chemical"]
    J = list(range(len(ccs)))

    mdl = Model(name="ReverseLogistics_MILP")
    mdl.context.cplex_parameters.timelimit = 60  # 60s time limit

    # ── Decision Variables ────────────────────────────────────────────────────
    # x[w,j] : kg of waste w sent to CC j
    x = {(w, j): mdl.continuous_var(lb=0, ub=ccs[j]["facilities"][w]["capacity"],
                                     name=f"x_{w}_{j}")
         for w in W for j in J}

    # y[j] : binary activation of CC j
    y = {j: mdl.binary_var(name=f"y_{j}") for j in J}

    # ── Objective Function ────────────────────────────────────────────────────
    # Revenue from solid & yarn
    revenue_terms = mdl.sum(
        ccs[j]["facilities"][w]["revenue_per_kg"] * x[w, j]
        for w in ["solid", "yarn"] for j in J
    )

    # Disposal fees for chemical
    disposal_terms = mdl.sum(
        ccs[j]["facilities"]["chemical"]["disposal_fee_per_kg"] * x["chemical", j]
        for j in J
    )

    # Transportation costs
    transport_terms = mdl.sum(
        transport[w] * ccs[j]["distance"] * x[w, j]
        for w in W for j in J
    )

    # Holding & preparation costs
    holding_terms = mdl.sum(
        holding[w] * x[w, j]
        for w in W for j in J
    )

    # ★ Fixed contract activation cost
    fixed_cost_terms = mdl.sum(
        ccs[j]["fixed_contract_cost"] * y[j]
        for j in J
    )

    # ★ Working capital / time cost
    # For solid & yarn: revenue is delayed T[w,j] days → opportunity cost = r × T × R × x
    # For chemical:     fee payment delayed T[w,j] days → also ties up cash → r × T × F × x
    wc_terms = mdl.sum(
        wc_rate
        * ccs[j]["facilities"][w]["process_days"]
        * (ccs[j]["facilities"][w]["revenue_per_kg"] if w in ["solid","yarn"]
           else ccs[j]["facilities"][w]["disposal_fee_per_kg"])
        * x[w, j]
        for w in W for j in J
    )

    # Environmental penalty (CO2)
    env_terms = mdl.sum(
        co2_pen * ccs[j]["facilities"][w]["co2_per_kg"] * x[w, j]
        for w in W for j in J
    )

    # Circularity bonus
    circ_terms = mdl.sum(
        circ_bonus * ccs[j]["facilities"][w]["circularity_rate"] * x[w, j]
        for w in W for j in J
    )

    # ETP net contribution (constant, not a decision variable)
    recycled_L  = treated_ww * etp["recycling_rate"]
    drained_L   = treated_ww * (1 - etp["recycling_rate"])
    etp_saving  = recycled_L * etp["reuse_value_per_L"]
    etp_drain   = drained_L  * etp["drain_cost_per_L"]
    etp_net     = etp_saving - etp_drain

    # Full objective
    mdl.maximize(
        revenue_terms
        - disposal_terms
        - transport_terms
        - holding_terms
        - fixed_cost_terms   # ★ NEW
        - wc_terms           # ★ NEW (time-value of money)
        - env_terms
        + circ_terms
        + etp_net            # constant
    )

    # ── Constraints ───────────────────────────────────────────────────────────

    # (1) All waste must be fully allocated
    for w in W:
        mdl.add_constraint(
            mdl.sum(x[w, j] for j in J) == wastes[w],
            ctname=f"full_allocation_{w}"
        )

    # (2) Capacity-linked to CC activation: x[w,j] ≤ CAP[w,j] · y[j]
    for w in W:
        for j in J:
            cap = ccs[j]["facilities"][w]["capacity"]
            mdl.add_constraint(
                x[w, j] <= cap * y[j],
                ctname=f"capacity_link_{w}_{j}"
            )

    # ── Solve ─────────────────────────────────────────────────────────────────
    print("\n  Calling IBM CPLEX solver...")
    sol = mdl.solve(log_output=False)

    if sol is None:
        print("\n  [ERROR] CPLEX found no feasible solution.")
        print("  Possible causes:")
        print("  - Total CC capacity < waste quantity for some type")
        print("  - Infeasible parameter combination")
        print("  Check your capacity inputs and retry.")
        sys.exit(1)

    # ── Extract Results ───────────────────────────────────────────────────────
    allocations = []
    for w in W:
        for j in J:
            qty = sol.get_value(x[w, j])
            if qty > 0.01:
                fac   = ccs[j]["facilities"][w]
                rev   = fac["revenue_per_kg"] * qty if w in ["solid","yarn"] else 0
                disp  = fac["disposal_fee_per_kg"] * qty if w == "chemical" else 0
                trans = transport[w] * ccs[j]["distance"] * qty
                hold  = holding[w] * qty
                wc    = wc_rate * fac["process_days"] * (fac["revenue_per_kg"] if w in ["solid","yarn"] else fac["disposal_fee_per_kg"]) * qty
                env   = co2_pen * fac["co2_per_kg"] * qty
                circ  = circ_bonus * fac["circularity_rate"] * qty
                net   = (rev - disp - trans - hold - wc - env + circ)
                allocations.append({
                    "waste_type": w,
                    "cc_name":    ccs[j]["name"],
                    "cc_index":   j,
                    "distance":   ccs[j]["distance"],
                    "quantity":   qty,
                    "revenue":    rev,
                    "disposal_fee": disp,
                    "transport":  trans,
                    "holding":    hold,
                    "wc_cost":    wc,
                    "env_penalty": env,
                    "circ_bonus": circ,
                    "net":        net,
                    "process_days": fac["process_days"],
                    "circularity_rate": fac["circularity_rate"],
                    "co2":        fac["co2_per_kg"] * qty,
                })

    active_ccs = [j for j in J if sol.get_value(y[j]) > 0.5]
    Z = sol.objective_value

    return {
        "allocations": allocations,
        "active_ccs": active_ccs,
        "Z": Z,
        "etp_net": etp_net,
        "etp_recycled": recycled_L,
        "etp_drained": drained_L,
        "etp_saving": etp_saving,
        "etp_drain": etp_drain,
        "model": mdl,
        "solution": sol,
    }


# ── SCIPY FALLBACK ────────────────────────────────────────────────────────────

def build_and_solve_scipy(data):
    """Fallback solver using scipy.optimize.milp (same formulation)."""
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    wastes     = data["wastes"]
    treated_ww = data["treated_ww"]
    holding    = data["holding"]
    transport  = data["transport"]
    etp        = data["etp"]
    wc_rate    = data["wc_rate"]
    co2_pen    = data["co2_penalty"]
    circ_bonus = data["circularity_bonus"]
    ccs        = data["ccs"]

    W = list(wastes.keys())
    J = list(range(len(ccs)))
    n_x = len(W) * len(J)
    n_y = len(J)
    n_vars = n_x + n_y

    def xi(w_idx, j): return w_idx * len(J) + j
    def yi(j): return n_x + j

    c = np.zeros(n_vars)
    for w_idx, w in enumerate(W):
        for j in J:
            fac  = ccs[j]["facilities"][w]
            dist = ccs[j]["distance"]
            t    = transport[w] * dist
            h    = holding[w]
            wc   = wc_rate * fac["process_days"] * (fac["revenue_per_kg"] if w in ["solid","yarn"] else fac["disposal_fee_per_kg"])
            env  = co2_pen * fac["co2_per_kg"]
            circ = circ_bonus * fac["circularity_rate"]
            if w in ["solid","yarn"]:
                net = fac["revenue_per_kg"] - t - h - wc - env + circ
            else:
                net = -(fac["disposal_fee_per_kg"] + t + h + wc + env) + circ
            c[xi(w_idx, j)] = -net  # negate for minimization

    for j in J:
        c[yi(j)] = ccs[j]["fixed_contract_cost"]  # fixed cost (already cost → add directly)

    integrality = np.zeros(n_vars)
    for j in J: integrality[yi(j)] = 1

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for j in J: ub[yi(j)] = 1.0
    for w_idx, w in enumerate(W):
        for j in J:
            ub[xi(w_idx, j)] = ccs[j]["facilities"][w]["capacity"]

    bounds = Bounds(lb, ub)

    A_rows, lb_rows, ub_rows = [], [], []
    for w_idx, w in enumerate(W):
        row = np.zeros(n_vars)
        for j in J: row[xi(w_idx, j)] = 1.0
        A_rows.append(row); lb_rows.append(wastes[w]); ub_rows.append(wastes[w])

    for w_idx, w in enumerate(W):
        for j in J:
            cap = ccs[j]["facilities"][w]["capacity"]
            row = np.zeros(n_vars)
            row[xi(w_idx, j)] = 1.0; row[yi(j)] = -cap
            A_rows.append(row); lb_rows.append(-np.inf); ub_rows.append(0.0)

    from scipy.optimize import LinearConstraint as LC
    result = milp(c, constraints=LC(np.array(A_rows), lb_rows, ub_rows),
                  integrality=integrality, bounds=bounds)

    if result.status != 0:
        print(f"\n[ERROR] scipy solver: {result.message}")
        sys.exit(1)

    x_sol = result.x
    recycled_L = treated_ww * etp["recycling_rate"]
    drained_L  = treated_ww * (1 - etp["recycling_rate"])
    etp_saving = recycled_L * etp["reuse_value_per_L"]
    etp_drain  = drained_L  * etp["drain_cost_per_L"]
    etp_net    = etp_saving - etp_drain

    allocations = []
    for w_idx, w in enumerate(W):
        for j in J:
            qty = x_sol[xi(w_idx, j)]
            if qty > 0.01:
                fac   = ccs[j]["facilities"][w]
                rev   = fac["revenue_per_kg"] * qty if w in ["solid","yarn"] else 0
                disp  = fac["disposal_fee_per_kg"] * qty if w == "chemical" else 0
                trans = transport[w] * ccs[j]["distance"] * qty
                hold  = holding[w] * qty
                wc    = wc_rate * fac["process_days"] * (fac["revenue_per_kg"] if w in ["solid","yarn"] else fac["disposal_fee_per_kg"]) * qty
                env   = co2_pen * fac["co2_per_kg"] * qty
                circ  = circ_bonus * fac["circularity_rate"] * qty
                net   = (rev - disp - trans - hold - wc - env + circ)
                allocations.append({
                    "waste_type": w, "cc_name": ccs[j]["name"], "cc_index": j,
                    "distance": ccs[j]["distance"], "quantity": qty,
                    "revenue": rev, "disposal_fee": disp, "transport": trans,
                    "holding": hold, "wc_cost": wc, "env_penalty": env,
                    "circ_bonus": circ, "net": net,
                    "process_days": fac["process_days"],
                    "circularity_rate": fac["circularity_rate"],
                    "co2": fac["co2_per_kg"] * qty,
                })

    active_ccs = [j for j in J if x_sol[yi(j)] > 0.5]
    Z = -result.fun + etp_net  # Negate back + add ETP constant

    return {
        "allocations": allocations, "active_ccs": active_ccs, "Z": Z,
        "etp_net": etp_net, "etp_recycled": recycled_L, "etp_drained": drained_L,
        "etp_saving": etp_saving, "etp_drain": etp_drain,
    }


# ── PRINT RESULTS ─────────────────────────────────────────────────────────────

def print_results(res, data):
    ccs     = data["ccs"]
    wastes  = data["wastes"]
    allocs  = res["allocations"]
    W       = list(wastes.keys())
    J       = list(range(len(ccs)))

    def f(v):  return f"{v:>12,.2f}"
    def fp(v): return f"{v*100:.1f}%"

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              OPTIMAL SOLUTION — CPLEX MILP                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Allocation plan
    print("\n  WASTE ALLOCATION PLAN\n  " + "─"*60)
    for w in W:
        w_allocs = [a for a in allocs if a["waste_type"] == w]
        print(f"\n  [{w.upper()} WASTE]  Total: {wastes[w]:,.2f} kg")
        for a in w_allocs:
            fc_share = ccs[a["cc_index"]]["fixed_contract_cost"]
            print(f"    → {a['cc_name']}")
            print(f"       Quantity:           {a['quantity']:>10,.2f} kg")
            if w in ["solid","yarn"]:
                print(f"       Revenue:            {a['revenue']:>10,.2f} BDT")
            else:
                print(f"       Disposal Fee:      -{a['disposal_fee']:>10,.2f} BDT")
            print(f"       Transport Cost:    -{a['transport']:>10,.2f} BDT")
            print(f"       Holding Cost:      -{a['holding']:>10,.2f} BDT")
            print(f"       Working Cap Cost:  -{a['wc_cost']:>10,.2f} BDT  (★ {a['process_days']} days × rate)")
            print(f"       Env Penalty:       -{a['env_penalty']:>10,.2f} BDT")
            print(f"       Circularity Bonus: +{a['circ_bonus']:>10,.2f} BDT  (rate={fp(a['circularity_rate'])})")
            print(f"       Net Contribution:   {a['net']:>10,.2f} BDT")
            print(f"       CO2 Emitted:        {a['co2']:>10,.2f} kg CO2-eq")
            print(f"       Process Days:       {a['process_days']} days")

    # Active CCs & Fixed Costs
    print(f"\n  ACTIVE COLLECTION CENTERS\n  " + "─"*60)
    total_fixed = 0
    for j in res["active_ccs"]:
        fc = ccs[j]["fixed_contract_cost"]
        total_fixed += fc
        print(f"    ✔  {ccs[j]['name']}   (fixed contract cost = {fc:,.2f} BDT)")

    # ETP
    print(f"\n  ETP — TREATED WASTEWATER\n  " + "─"*60)
    print(f"    Total Treated:    {data['treated_ww']:>12,.2f} L")
    print(f"    Recycled:         {res['etp_recycled']:>12,.2f} L  →  Savings: {res['etp_saving']:,.2f} BDT")
    print(f"    Drained:          {res['etp_drained']:>12,.2f} L  →  Cost:    {res['etp_drain']:,.2f} BDT")
    print(f"    Net ETP Value:    {res['etp_net']:>12,.2f} BDT")

    # Financial Summary
    total_rev   = sum(a["revenue"]     for a in allocs)
    total_disp  = sum(a["disposal_fee"] for a in allocs)
    total_trans = sum(a["transport"]   for a in allocs)
    total_hold  = sum(a["holding"]     for a in allocs)
    total_wc    = sum(a["wc_cost"]     for a in allocs)
    total_env   = sum(a["env_penalty"] for a in allocs)
    total_circ  = sum(a["circ_bonus"]  for a in allocs)
    total_co2   = sum(a["co2"]         for a in allocs)
    total_qty   = sum(a["quantity"]    for a in allocs)
    avg_circ    = sum(a["circularity_rate"]*a["quantity"] for a in allocs) / total_qty

    print(f"\n  FINANCIAL SUMMARY\n  " + "═"*60)
    print(f"  {'Revenue (Solid + Yarn)':45s}  +{f(total_rev)} BDT")
    print(f"  {'Disposal Fees (Chemical)':45s}  -{f(total_disp)} BDT")
    print(f"  {'Transportation Costs':45s}  -{f(total_trans)} BDT")
    print(f"  {'Holding & Preparation Costs':45s}  -{f(total_hold)} BDT")
    print(f"  {'Fixed Contract Costs (★ activation)':45s}  -{f(total_fixed)} BDT")
    print(f"  {'Working Capital Costs (★ time-delay)':45s}  -{f(total_wc)} BDT")
    print(f"  {'Environmental Penalties (CO2)':45s}  -{f(total_env)} BDT")
    print(f"  {'Circularity Bonuses':45s}  +{f(total_circ)} BDT")
    print(f"  {'ETP Net Contribution':45s}  +{f(res['etp_net'])} BDT")
    print(f"  {'─'*60}")
    sign = "+" if res["Z"] >= 0 else ""
    print(f"  {'OPTIMAL OBJECTIVE VALUE  Z  (MILP)':45s}   {sign}{res['Z']:,.2f} BDT")
    print(f"\n  PERFORMANCE METRICS\n  " + "─"*60)
    print(f"  {'Weighted Avg Circularity Rate':40s}  {avg_circ*100:.2f}%")
    print(f"  {'Total CO2 Emissions':40s}  {total_co2:,.2f} kg CO2-eq")
    print(f"  {'Number of Active CCs':40s}  {len(res['active_ccs'])}")
    print(f"  {'Total Waste Processed':40s}  {total_qty:,.2f} kg")
    print("\n" + "═"*65)
    print("  Solver: IBM CPLEX via DOcplex  |  Model: MILP")
    print("  Thesis: Circular Supply Chain Reverse Logistics Optimization")
    print("═"*65 + "\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    data = collect_inputs()

    header("SOLVING THE MILP MODEL")

    if USE_SCIPY_FALLBACK:
        print("\n  [INFO] Using scipy fallback solver (CPLEX not configured).")
        res = build_and_solve_scipy(data)
    elif check_docplex():
        print(f"\n  [INFO] DOcplex found. {'CPLEX detected.' if check_cplex() else 'Using DOcplex cloud/community.'}")
        res = build_and_solve_cplex(data)
    else:
        print("\n  [WARNING] DOcplex not found. Falling back to scipy.optimize.milp.")
        print("  To install DOcplex: pip install docplex cplex")
        res = build_and_solve_scipy(data)

    print_results(res, data)

    # Export to CSV
    import csv, os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milp_results.csv")
    with open(out_path, "w", newline="") as f:
        if res["allocations"]:
            writer = csv.DictWriter(f, fieldnames=res["allocations"][0].keys())
            writer.writeheader()
            writer.writerows(res["allocations"])
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
