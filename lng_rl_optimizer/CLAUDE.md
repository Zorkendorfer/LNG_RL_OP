# LNG Terminal RL Energy Optimizer — Session Context

## What this is
Safe RL agent that minimizes electricity cost for LNG terminal operations.
Novel vs existing work (RALT-DT 2025) in three ways:
1. Optimizes electricity COST vs NordPool prices, not just energy volume
2. LNG composition (methane/ethane/propane) is a dynamic state variable
3. Physics-informed neural network surrogate instead of black-box NN

Target: Klaipėda FSRU Independence (publicly documented 170,000 m³ FSRU).

## Stack
Python 3.11 | CoolProp | PyTorch | Gymnasium | Stable-Baselines3 | MLflow

## Pipeline
NordPool prices → price_forecaster.py
Physics simulator → surrogate training data → TerminalPINN
TerminalPINN + price forecaster → LNGTerminalEnv (Gymnasium)
LNGTerminalEnv → PPO agent (SB3) → annual cost savings vs rule-based baseline

## State vector (44 dims)
[0]   fill_fraction          [1]   tank_pressure_norm
[2]   x_methane              [3]   x_ethane
[4]   ambient_temp_norm      [5]   seawater_temp_norm
[6]   sea_state              [7]   current_price_norm
[8:32] price_forecast_24h   [32:34] hour sin/cos
[35:37] compressor_loads    [38:39] n_pumps, n_vap
[40:43] wear tracking

## Action space (MultiDiscrete [4,4,3,4,4])
comp_0 load, comp_1 load, comp_2 load, n_hp_pumps, n_vaporizers

## Reward
-electricity_cost_eur_h - carbon_cost - pressure_penalties - bog_penalties - wear_cost

## Key source files
src/physics/terminal_simulator.py  — ODE physics model (surrogate training)
src/physics/bog_model.py           — BOG generation (FSRU-aware)
src/surrogate/pinn.py              — TerminalPINN with physics constraints
src/market/nordpool_fetcher.py     — NordPool Baltic price data
src/environment/lng_terminal_env.py — Gymnasium env (core)
src/environment/reward.py          — cost + penalties + carbon
src/agent/baseline_agents.py       — rule-based + price-heuristic baselines

## Paper contribution framing
Primary result: annual electricity cost savings (EUR) vs rule-based baseline.
Secondary results: constraint violation rate, price-load correlation (novelty proof).
Ablation: show each of 3 novelties independently contributes to savings.

## Current status
Phase 0 complete — project structure created.
Next: download NordPool price data, then validate physics model.

## Next task
1. Set ENTSOE_TOKEN env var and run: python scripts/download_nordpool_data.py
2. Validate physics: python scripts/check_env.py
3. Generate surrogate training data: python scripts/generate_surrogate_data.py
