"""Day-by-day inventory simulation (lost sales, no backorders)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    on_hand_end: np.ndarray  # (T, N)
    sold: np.ndarray
    lost: np.ndarray
    orders: np.ndarray       # quantity ordered at the end of each period
    n_orders: np.ndarray     # 1 where an order was placed


def simulate_base_stock(
    demand: np.ndarray,
    order_up_to: np.ndarray,
    lead_time_days: int,
    *,
    reorder_point: np.ndarray | None = None,
    batch_size: int = 1,
    min_order_qty: float = 0.0,
    delay_prob: float = 0.0,
    delay_max: int = 0,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """demand: (T, N); order_up_to: (T+1, N) as returned by ``base_stock_components`` (sum).

    reorder_point (T+1, N): if given the policy is (s, S) - order up to S only when the inventory position is
    at or below s. batch_size / min_order_qty round orders up. With ``delay_prob > 0`` each order may arrive
    1..delay_max periods late (the same ``rng`` seed gives every policy the same delays).
    """
    T, N = demand.shape
    L, D = lead_time_days, int(delay_max if delay_prob > 0 else 0)
    rng = rng or np.random.default_rng(0)
    pipeline = np.zeros((T + L + D + 3, N))  # pipeline[t] = units arriving at the start of period t
    on_hand = order_up_to[0].copy()          # start fully stocked
    on_hand_end, sold, lost = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    orders, n_orders = np.zeros((T, N)), np.zeros((T, N))

    for t in range(T):
        on_hand = on_hand + pipeline[t]
        s = np.minimum(on_hand, demand[t])
        sold[t] = s
        lost[t] = demand[t] - s
        on_hand = on_hand - s
        on_hand_end[t] = on_hand
        position = on_hand + pipeline[t + 1 :].sum(axis=0)
        gap = order_up_to[t + 1] - position
        q = np.maximum(0.0, gap)
        if reorder_point is not None:
            q = np.where(position <= reorder_point[t + 1], q, 0.0)
        if batch_size > 1:
            q = np.ceil(q / batch_size) * batch_size
        if min_order_qty > 0:
            q = np.where((q > 0) & (q < min_order_qty), min_order_qty, q)
        delay = np.where(rng.random(N) < delay_prob, rng.integers(1, max(D, 1) + 1, N), 0) if D > 0 else np.zeros(N, int)
        arrive = t + 1 + L + delay
        pipeline[arrive, np.arange(N)] += q
        orders[t] = q
        n_orders[t] = q > 0
    return SimResult(on_hand_end, sold, lost, orders, n_orders)


def summarize(sim: SimResult, demand: np.ndarray, holding_cost: np.ndarray, stockout_cost: np.ndarray,
              fixed_order_cost: float = 0.0) -> dict:
    """holding_cost: per unit per period; stockout_cost: per lost unit (arrays of length N)."""
    total_demand = float(demand.sum())
    holding = float((sim.on_hand_end * holding_cost[None, :]).sum())
    stockout = float((sim.lost * stockout_cost[None, :]).sum())
    n = float(sim.n_orders.sum())
    return {
        "fill_rate": float(sim.sold.sum() / total_demand) if total_demand else float("nan"),
        "stockout_rate": float((sim.lost > 0).mean()),
        "lost_units": float(sim.lost.sum()),
        "avg_on_hand_units": float(sim.on_hand_end.mean()),
        "n_orders": n,
        "holding_cost": holding,
        "stockout_cost": stockout,
        "order_cost": n * fixed_order_cost,
        "total_cost": holding + stockout + n * fixed_order_cost,
    }
