# Predictive Waste Classification and Reverse Logistics Optimization Using Machine Learning in a Circular Supply Chain

**Case Study: Thread Dyeing Company**

This repository contains the modeling code for my B.Sc. thesis at BUET, which combines
machine-learning-based waste prediction with a mixed-integer linear program (MILP) for
reverse-logistics network design in a textile (thread-dyeing) circular supply chain.

## Overview

The factory generates three waste streams — **solid**, **yarn**, and **chemical** — plus
treated wastewater (partly reused on-site via the ETP, remainder drained). The project has
two connected layers:

1. **Prediction layer** — XGBoost and Random Forest models forecast daily waste generation
   per stream from production and process parameters (energy/water use, dye/salt/alkali
   dosing, cycle time, etc.).
2. **Optimization layer** — A MILP (solved via IBM CPLEX / DOcplex) decides which
   Collection Centers (CCs) to activate and how to route each waste stream from the
   factory to CCs and on to processing facilities, minimizing total cost (transport,
   holding, disposal) net of recovery value and environmental penalties/bonuses.

A **scenario/sensitivity analysis** propagates the ML models' prediction error (±1.96×RMSE
per waste stream) into the MILP to test whether the optimal network decision (which CCs
are activated) is stable under realistic forecast uncertainty — addressing the common
"deterministic optimizer fed by uncertain ML" critique.

An interactive **React dashboard** reimplements the allocation logic in pure JavaScript so
the model can be explored in a browser without any Python/CPLEX setup.

## Repository structure

```
├── notebooks/
│   ├── 01_xgboost_waste_prediction.ipynb                  # XGBoost model: waste stream forecasting
│   ├── 02_randomforest_waste_prediction.ipynb             # Random Forest model: waste stream forecasting
│   ├── 03_reverse_logistics_milp.ipynb                    # MILP network design (notebook version)
│   └── 04_reverse_logistics_milp_with_xgboost_inputs.ipynb # MILP driven by XGBoost predictions end-to-end
├── src/
│   ├── reverse_logistics_milp.py    # Standalone MILP formulation (DOcplex)
│   └── scenario_analysis.py         # Sensitivity analysis: ±1.96*RMSE demand scenarios vs. MILP stability
├── dashboard/
│   └── ReverseLogisticsDashboard.jsx # Interactive browser demo (React + Recharts), no CPLEX required
├── data/
│   └── textile_dyeing_dataset.xlsx  # Process/production dataset used to train the prediction models
├── requirements.txt
├── LICENSE
└── .gitignore
```

## MILP formulation (summary)

**Sets:** `W` = waste types {solid, yarn, chemical}; `J` = candidate Collection Centers

**Decision variables:**
- `x[w,j] ≥ 0` — kg of waste type `w` routed to Collection Center `j`
- `y[j] ∈ {0,1}` — 1 if Collection Center `j` is activated/contracted

**Objective:** minimize total network cost — transport + holding + disposal, net of
recovery revenue, CO₂ penalties, and circularity bonuses — subject to demand satisfaction,
CC capacity, and activation-linking constraints.

Full formulation and constraint set are documented inline in `src/reverse_logistics_milp.py`.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_xgboost_waste_prediction.ipynb
```

> **Note on CPLEX:** `src/reverse_logistics_milp.py`, `src/scenario_analysis.py`, and the
> MILP notebooks require IBM CPLEX (via `docplex`). The free
> [Community Edition](https://www.ibm.com/products/ilog-cplex-optimization-studio) works
> for small problem instances like this one; a full license is only needed for larger
> networks. If you just want to explore the optimization logic without installing CPLEX,
> open `dashboard/ReverseLogisticsDashboard.jsx` instead — it's a dependency-free
> reimplementation in JavaScript.

## Running the dashboard

The dashboard is a single React component (uses `recharts` for the visualizations). Drop
it into any React + Tailwind project, or deploy it standalone via Vite/CodeSandbox/GitHub
Pages for a link you can share directly.

## Data note

`data/textile_dyeing_dataset.xlsx` contains process-level production data (732 daily
records; 39 fields covering production volume, energy/water consumption, dye/chemical
dosing, and resulting waste streams) used to train the prediction models. If any of this
reflects proprietary process data from the case-study company, confirm sharing terms
before treating this repository as fully public.

## Citation

If you build on this work, please cite:

> Halder, A. *Predictive Waste Classification and Reverse Logistics Optimization Using
> Machine Learning in a Circular Supply Chain: A Case Study of Thread Dyeing Company.*
> B.Sc. Thesis, Bangladesh University of Engineering and Technology (BUET).

## License

MIT — see [LICENSE](LICENSE).
