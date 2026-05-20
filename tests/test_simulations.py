from __future__ import annotations

import unittest

from trans_vfl.simulations import (
    EHRSimulationConfig,
    GenomicSimulationConfig,
    run_ehr_simulation,
    run_genomic_simulation,
)


class SimulationTests(unittest.TestCase):
    def test_ehr_simulation_returns_table_metrics(self) -> None:
        config = EHRSimulationConfig(n_sites=3, n_patients=300, replications=2)
        result = run_ehr_simulation(config, seed=1)

        self.assertIn("accuracy", result)
        self.assertEqual(result["avg_features_selected"], float(config.selected_features))
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_genomic_simulation_reports_gene_counts(self) -> None:
        config = GenomicSimulationConfig(n_sites=2, n_patients=120, replications=2)
        result = run_genomic_simulation(config, seed=1)

        self.assertEqual(result["avg_pathways_selected"], float(config.selected_pathways))
        self.assertEqual(
            result["avg_genes_selected"],
            float(config.selected_pathways * config.genes_per_pathway),
        )


if __name__ == "__main__":
    unittest.main()
