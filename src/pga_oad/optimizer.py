from __future__ import annotations
from typing import Optional

import pulp

from .models import OaDPick, Prediction, Tournament


class OneAndDoneOptimizer:
    def __init__(self, used_players: Optional[set[int]] = None):
        self.used_players: set[int] = used_players or set()

    def optimize(
        self,
        tournaments: list[Tournament],
        predictions: dict[int, list[Prediction]],
    ) -> list[OaDPick]:
        """
        Solve the One and Done assignment problem.

        Args:
            tournaments: Upcoming tournaments to assign picks to.
            predictions: Map of event_id -> list of Prediction for players in that field.

        Returns:
            List of OaDPick, one per tournament, sorted by event order.
        """
        # Collect all unique players across all tournament fields
        all_players: dict[int, str] = {}
        for preds in predictions.values():
            for p in preds:
                if p.dg_id not in self.used_players:
                    all_players[p.dg_id] = p.player_name

        if not all_players:
            raise ValueError("No available players after applying used_players filter.")

        player_ids = list(all_players.keys())
        event_ids = [t.event_id for t in tournaments]

        # Build expected value lookup: ev[(dg_id, event_id)]
        ev: dict[tuple[int, int], float] = {}
        for event_id, preds in predictions.items():
            for p in preds:
                if p.dg_id not in self.used_players:
                    ev[(p.dg_id, event_id)] = p.expected_value

        prob = pulp.LpProblem("OneAndDone", pulp.LpMaximize)

        # Binary decision variables: x[player, event] = 1 if picked
        x = pulp.LpVariable.dicts(
            "x",
            [(p, e) for p in player_ids for e in event_ids],
            cat="Binary",
        )

        # Zero out variables for players not in a tournament's field
        for p in player_ids:
            for e in event_ids:
                if (p, e) not in ev:
                    prob += x[(p, e)] == 0

        # Objective: maximize total expected value
        prob += pulp.lpSum(
            ev.get((p, e), 0) * x[(p, e)]
            for p in player_ids
            for e in event_ids
        )

        # Constraint: each player used at most once across all tournaments
        for p in player_ids:
            prob += pulp.lpSum(x[(p, e)] for e in event_ids) <= 1

        # Constraint: exactly one player picked per tournament
        for e in event_ids:
            prob += pulp.lpSum(x[(p, e)] for p in player_ids) == 1

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if prob.status != 1:
            raise RuntimeError(f"Optimizer failed with status: {pulp.LpStatus[prob.status]}")

        # Extract results
        picks: list[OaDPick] = []
        event_map = {t.event_id: t for t in tournaments}

        for e in event_ids:
            for p in player_ids:
                if pulp.value(x[(p, e)]) == 1:
                    pred_ev = ev.get((p, e), 0.0)
                    # Find the full prediction for win_prob
                    win_prob = 0.0
                    for pred in predictions.get(e, []):
                        if pred.dg_id == p:
                            win_prob = pred.win_prob
                            break
                    picks.append(
                        OaDPick(
                            player_name=all_players[p],
                            dg_id=p,
                            event_name=event_map[e].event_name,
                            event_id=e,
                            win_prob=win_prob,
                            expected_value=round(pred_ev, 4),
                        )
                    )
                    break

        # Sort by the order tournaments were provided
        event_order = {e: i for i, e in enumerate(event_ids)}
        picks.sort(key=lambda pick: event_order[pick.event_id])
        return picks
