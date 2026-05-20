from __future__ import annotations

import unittest

import numpy as np

from trans_vfl.core import (
    LocalTransVFLModel,
    feature_selection_metrics,
    local_feature_selection,
    proximal_group_lasso_rows,
    select_embedding_components,
)


class CoreTests(unittest.TestCase):
    def test_proximal_group_lasso_rows_shrinks_by_row(self) -> None:
        weights = np.array([[3.0, 4.0], [0.2, 0.0], [0.0, 0.0]])
        shrunk = proximal_group_lasso_rows(weights, step_size=1.0)
        self.assertAlmostEqual(np.linalg.norm(shrunk[0]), 4.0)
        self.assertTrue(np.allclose(shrunk[1], 0.0))
        self.assertTrue(np.allclose(shrunk[2], 0.0))

    def test_select_embedding_components_finds_correlated_dimension(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.binomial(1, 0.5, size=200).astype(float)
        embeddings = np.column_stack(
            [
                rng.normal(size=200),
                y + rng.normal(0.0, 0.05, size=200),
                rng.normal(size=200),
            ]
        )
        selected = select_embedding_components([embeddings], y, top_k=1)
        self.assertEqual(selected[0].tolist(), [1])

    def test_local_feature_selection_prunes_weak_rows(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240, 6))
        teacher = LocalTransVFLModel.random(6, d_select=5, d_hidden=4, d_embed=3, rng=rng)
        teacher.w_select[:2] = rng.normal(0.0, 0.8, size=(2, 5))
        teacher.w_select[2:] = rng.normal(0.0, 0.005, size=(4, 5))

        result = local_feature_selection(
            x,
            teacher,
            [0, 1, 2],
            lambda_value=0.03,
            epochs=90,
            batch_size=80,
            lr=0.03,
            threshold=1e-3,
            seed=4,
        )

        self.assertTrue(set([0, 1]).issubset(set(result.kept_features.tolist())))
        self.assertGreaterEqual(len(set([2, 3, 4, 5]) & set(result.pruned_features.tolist())), 2)

    def test_feature_selection_metrics(self) -> None:
        metrics = feature_selection_metrics([0, 1, 3], [0, 1, 2], total_features=5)
        self.assertEqual(metrics["tp"], 2.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertAlmostEqual(metrics["tpr"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["fdr"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
