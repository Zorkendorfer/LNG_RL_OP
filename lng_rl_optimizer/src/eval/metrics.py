import numpy as np
import pandas as pd


def annual_cost_savings(
    rl_costs_eur: list,
    baseline_costs_eur: list,
) -> dict:
    """Compare annual electricity costs: RL agent vs rule-based baseline."""
    rl_total   = sum(rl_costs_eur)
    base_total = sum(baseline_costs_eur)
    savings    = base_total - rl_total
    savings_pct = savings / (base_total + 1e-8) * 100

    return {
        "rl_annual_cost_eur":       rl_total,
        "baseline_annual_cost_eur": base_total,
        "savings_eur":              savings,
        "savings_pct":              savings_pct,
    }


def constraint_violation_rate(episode_infos: list) -> dict:
    """Safety metric: fraction of timesteps violating each constraint."""
    n = len(episode_infos)
    if n == 0:
        return {}
    return {
        "overpressure_rate":  sum("overpressure"  in i for i in episode_infos) / n,
        "underpressure_rate": sum("underpressure" in i for i in episode_infos) / n,
        "low_inventory_rate": sum("low_inventory" in i for i in episode_infos) / n,
        "bog_flaring_rate":   sum("bog_flaring"   in i for i in episode_infos) / n,
    }


def price_correlation_analysis(
    actions: np.ndarray,
    prices: np.ndarray,
) -> float:
    """
    Key novelty verification: does the agent shift load to low-price periods?
    Negative correlation = good (less power when prices are high).
    """
    comp_load = actions[:, 0] + actions[:, 1]
    return float(np.corrcoef(comp_load, prices)[0, 1])


def evaluate_agent(env, agent, n_episodes: int = 5) -> dict:
    """Run agent for n_episodes and compute aggregate metrics."""
    all_costs   = []
    all_infos   = []
    all_actions = []
    all_prices  = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_cost = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_cost += info.get("cost_eur_h", 0)
            all_infos.append(info)
            all_actions.append(list(action))
            all_prices.append(obs[7] * 300.0)
        all_costs.append(ep_cost)
        print(f"Episode {ep+1}: cost = €{ep_cost:,.0f}")

    return {
        "mean_annual_cost":   np.mean(all_costs),
        "std_annual_cost":    np.std(all_costs),
        "violation_rates":    constraint_violation_rate(all_infos),
        "price_correlation":  price_correlation_analysis(
            np.array(all_actions), np.array(all_prices)
        ),
    }
