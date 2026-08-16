import unittest

from context_experiments import (
    Evidence,
    pack_position_aware,
    run_lost_middle_experiment,
)


class ContextExperimentTests(unittest.TestCase):
    def test_position_aware_pack_puts_high_value_evidence_at_edges(self) -> None:
        evidence = [Evidence(str(index), f"evidence-{index}", float(index)) for index in range(5)]
        packed = pack_position_aware(evidence)
        self.assertIn(packed[0].evidence_id, {"4", "3"})
        self.assertIn(packed[-1].evidence_id, {"4", "3"})

    def test_experiment_reports_middle_drop_for_position_sensitive_reader(self) -> None:
        def reader(context: str, question: str) -> bool:
            lines = context.splitlines()
            return "Needle fact" in lines[0] or "Needle fact" in lines[-1]

        metrics = run_lost_middle_experiment(reader)
        self.assertEqual(metrics.accuracy_by_position[0], 1)
        self.assertEqual(metrics.accuracy_by_position[4], 0)
        self.assertGreater(metrics.middle_drop, 0)


if __name__ == "__main__":
    unittest.main()
