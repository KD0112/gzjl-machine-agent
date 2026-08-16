"""Context-placement experiments for the lost-in-the-middle failure mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    text: str
    relevance: float
    trusted: bool = False


@dataclass(frozen=True, slots=True)
class PlacementResult:
    strategy: str
    position: int
    context: str
    target_id: str


@dataclass(frozen=True, slots=True)
class LostMiddleMetrics:
    accuracy_by_position: dict[int, float]
    edge_accuracy: float
    middle_accuracy: float
    middle_drop: float


def pack_position_aware(evidence: Sequence[Evidence], *, strategy: str = "position_aware") -> list[Evidence]:
    """Place high-value evidence at the edges and low-value noise in the middle."""
    ranked = sorted(evidence, key=lambda item: (-item.relevance, item.evidence_id))
    if strategy == "ranked":
        return ranked
    if strategy not in {"position_aware", "ranked"}:
        raise ValueError("strategy must be ranked or position_aware")
    placed: list[Evidence | None] = [None] * len(ranked)
    left, right = 0, len(ranked) - 1
    for index, item in enumerate(ranked):
        if index % 2 == 0:
            placed[left] = item
            left += 1
        else:
            placed[right] = item
            right -= 1
    return [item for item in placed if item is not None]


def make_lost_middle_case(*, item_count: int = 9, target_position: int) -> PlacementResult:
    """Build one needle-in-context case; target_position is zero-based."""
    if item_count < 3 or not 0 <= target_position < item_count:
        raise ValueError("item_count must be >= 3 and target_position must be in range")
    target = Evidence("target", "Needle fact: delivery deadline is 2026-09-30.", 1.0, True)
    filler = [Evidence(f"noise-{i}", f"Unrelated catalogue paragraph {i}.", 0.05) for i in range(item_count - 1)]
    items = filler[:target_position] + [target] + filler[target_position:]
    return PlacementResult("manual_position", target_position, "\n".join(item.text for item in items), target.evidence_id)


def run_lost_middle_experiment(
    reader: Callable[[str, str], bool],
    *,
    item_count: int = 9,
    positions: Sequence[int] | None = None,
) -> LostMiddleMetrics:
    """Run the same question across positions and quantify the middle drop.

    ``reader`` can wrap a real model call.  It returns whether the answer
    contains the expected needle fact, allowing a cheap offline harness and a
    production-model evaluation to share the same metric code.
    """
    positions = tuple(positions or (0, 1, item_count // 2, item_count - 2, item_count - 1))
    accuracies: dict[int, float] = {}
    for position in positions:
        case = make_lost_middle_case(item_count=item_count, target_position=position)
        accuracies[position] = 1.0 if reader(case.context, "What is the delivery deadline?") else 0.0
    edge_positions = [position for position in positions if position in {0, item_count - 1}]
    middle_positions = [position for position in positions if position == item_count // 2]
    edge = sum(accuracies[position] for position in edge_positions) / len(edge_positions) if edge_positions else 0.0
    middle = sum(accuracies[position] for position in middle_positions) / len(middle_positions) if middle_positions else 0.0
    return LostMiddleMetrics(accuracies, edge, middle, edge - middle)


def position_sweep_summary(metrics: LostMiddleMetrics) -> dict[str, float]:
    return {
        "edge_accuracy": round(metrics.edge_accuracy, 4),
        "middle_accuracy": round(metrics.middle_accuracy, 4),
        "middle_drop": round(metrics.middle_drop, 4),
        "max_position": float(max(metrics.accuracy_by_position, key=metrics.accuracy_by_position.get)),
    }
