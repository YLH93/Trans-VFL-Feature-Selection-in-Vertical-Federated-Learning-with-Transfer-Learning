"""Core NumPy implementation of the Trans-VFL workflow.

The PDF defines the local selection matrix as theta_select in
R^(d_base x d_select). This module keeps the same orientation: every row is one
learned feature from the frozen base, and Stage 3 applies group lasso across
those rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


def _as_2d_float(name: str, value: Array) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def _as_binary_targets(y: Array) -> Array:
    targets = np.asarray(y, dtype=float).reshape(-1)
    unique = np.unique(targets)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError("y must contain binary labels encoded as 0/1")
    return targets


def _sigmoid(x: Array) -> Array:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _relu(x: Array) -> Array:
    return np.maximum(x, 0.0)


@dataclass
class LocalTransVFLModel:
    """Local site model: selection layer plus two-layer embedding head."""

    w_select: Array
    b_select: Array
    w_head1: Array
    b_head1: Array
    w_head2: Array
    b_head2: Array

    @classmethod
    def random(
        cls,
        d_base: int,
        d_select: int = 32,
        d_hidden: int = 24,
        d_embed: int = 16,
        *,
        rng: np.random.Generator | None = None,
        scale: float = 0.12,
    ) -> "LocalTransVFLModel":
        rng = rng or np.random.default_rng()
        return cls(
            w_select=rng.normal(0.0, scale, size=(d_base, d_select)),
            b_select=np.zeros(d_select),
            w_head1=rng.normal(0.0, scale, size=(d_select, d_hidden)),
            b_head1=np.zeros(d_hidden),
            w_head2=rng.normal(0.0, scale, size=(d_hidden, d_embed)),
            b_head2=np.zeros(d_embed),
        )

    @property
    def d_base(self) -> int:
        return int(self.w_select.shape[0])

    @property
    def d_select(self) -> int:
        return int(self.w_select.shape[1])

    @property
    def d_embed(self) -> int:
        return int(self.w_head2.shape[1])

    def copy(self) -> "LocalTransVFLModel":
        return LocalTransVFLModel(
            w_select=self.w_select.copy(),
            b_select=self.b_select.copy(),
            w_head1=self.w_head1.copy(),
            b_head1=self.b_head1.copy(),
            w_head2=self.w_head2.copy(),
            b_head2=self.b_head2.copy(),
        )

    def forward(self, z: Array, *, return_cache: bool = False) -> Array | tuple[Array, dict[str, Array]]:
        z = _as_2d_float("z", z)
        if z.shape[1] != self.d_base:
            raise ValueError(f"z has {z.shape[1]} features, expected {self.d_base}")

        select_pre = z @ self.w_select + self.b_select
        selected = _relu(select_pre)
        head_pre = selected @ self.w_head1 + self.b_head1
        hidden = _relu(head_pre)
        embedding = hidden @ self.w_head2 + self.b_head2

        if not return_cache:
            return embedding

        return embedding, {
            "z": z,
            "select_pre": select_pre,
            "selected": selected,
            "head_pre": head_pre,
            "hidden": hidden,
        }

    def feature_norms(self) -> Array:
        return np.linalg.norm(self.w_select, axis=1)

    def hard_prune(self, threshold: float) -> tuple[Array, Array]:
        norms = self.feature_norms()
        pruned = np.flatnonzero(norms <= threshold)
        kept = np.flatnonzero(norms > threshold)
        self.w_select[pruned, :] = 0.0
        return kept, pruned


@dataclass
class ServerFusionModel:
    """Simple server-side binary classifier over concatenated site embeddings."""

    weights: Array
    bias: float = 0.0

    @classmethod
    def random(
        cls,
        n_sites: int,
        d_embed: int,
        *,
        rng: np.random.Generator | None = None,
        scale: float = 0.05,
    ) -> "ServerFusionModel":
        rng = rng or np.random.default_rng()
        return cls(weights=rng.normal(0.0, scale, size=n_sites * d_embed), bias=0.0)

    def predict_logits(self, concatenated_embeddings: Array) -> Array:
        embeddings = _as_2d_float("concatenated_embeddings", concatenated_embeddings)
        return embeddings @ self.weights + self.bias

    def predict_proba(self, concatenated_embeddings: Array) -> Array:
        return _sigmoid(self.predict_logits(concatenated_embeddings))


@dataclass
class Stage1Result:
    local_models: list[LocalTransVFLModel]
    server_model: ServerFusionModel
    losses: list[float]


@dataclass
class Stage3Result:
    model: LocalTransVFLModel
    kept_features: Array
    pruned_features: Array
    feature_norms: Array
    losses: list[float]
    reconstruction_loss: float
    sparsity: float
    lambda_value: float
    selected_embedding_indices: Array


@dataclass
class LambdaTuningResult:
    best_lambda: float
    best_result: Stage3Result
    scores: dict[float, float]


def group_lasso_penalty(w_select: Array) -> float:
    """Return the L2,1 penalty used for learned-feature pruning."""

    weights = _as_2d_float("w_select", w_select)
    return float(np.linalg.norm(weights, axis=1).sum())


def proximal_group_lasso_rows(w_select: Array, step_size: float) -> Array:
    """Apply row-wise group-lasso shrinkage to a selection matrix."""

    weights = _as_2d_float("w_select", w_select)
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    scale = np.maximum(0.0, 1.0 - step_size / np.maximum(norms, 1e-12))
    return weights * scale


def _backward_local(model: LocalTransVFLModel, cache: dict[str, Array], grad_embedding: Array) -> dict[str, Array]:
    grad_w_head2 = cache["hidden"].T @ grad_embedding
    grad_b_head2 = grad_embedding.sum(axis=0)

    grad_hidden = grad_embedding @ model.w_head2.T
    grad_head_pre = grad_hidden * (cache["head_pre"] > 0.0)
    grad_w_head1 = cache["selected"].T @ grad_head_pre
    grad_b_head1 = grad_head_pre.sum(axis=0)

    grad_selected = grad_head_pre @ model.w_head1.T
    grad_select_pre = grad_selected * (cache["select_pre"] > 0.0)
    grad_w_select = cache["z"].T @ grad_select_pre
    grad_b_select = grad_select_pre.sum(axis=0)

    return {
        "w_select": grad_w_select,
        "b_select": grad_b_select,
        "w_head1": grad_w_head1,
        "b_head1": grad_b_head1,
        "w_head2": grad_w_head2,
        "b_head2": grad_b_head2,
    }


def _apply_local_gradients(model: LocalTransVFLModel, gradients: dict[str, Array], lr: float) -> None:
    model.w_select -= lr * gradients["w_select"]
    model.b_select -= lr * gradients["b_select"]
    model.w_head1 -= lr * gradients["w_head1"]
    model.b_head1 -= lr * gradients["b_head1"]
    model.w_head2 -= lr * gradients["w_head2"]
    model.b_head2 -= lr * gradients["b_head2"]


def collaborative_pretrain(
    site_features: Sequence[Array],
    y: Array,
    *,
    local_models: Sequence[LocalTransVFLModel] | None = None,
    d_select: int = 32,
    d_hidden: int = 24,
    d_embed: int = 16,
    epochs: int = 20,
    batch_size: int = 256,
    lr_local: float = 0.01,
    lr_server: float = 0.05,
    seed: int = 0,
) -> Stage1Result:
    """Stage 1: collaborative pretraining with frozen base features.

    The inputs are already the frozen base outputs z_m. Sites compute embeddings,
    the server trains a fusion classifier, and gradients are sent back to update
    only the local selection/head parameters.
    """

    if not site_features:
        raise ValueError("site_features must contain at least one site")

    features = [_as_2d_float(f"site_features[{idx}]", x) for idx, x in enumerate(site_features)]
    targets = _as_binary_targets(y)
    n_samples = targets.shape[0]
    if any(x.shape[0] != n_samples for x in features):
        raise ValueError("all site feature matrices must have the same row count as y")

    rng = np.random.default_rng(seed)
    if local_models is None:
        models = [
            LocalTransVFLModel.random(
                x.shape[1],
                d_select=d_select,
                d_hidden=d_hidden,
                d_embed=d_embed,
                rng=rng,
            )
            for x in features
        ]
    else:
        models = [model.copy() for model in local_models]
        d_embed = models[0].d_embed
        if any(model.d_embed != d_embed for model in models):
            raise ValueError("all local models must use the same embedding dimension")

    server = ServerFusionModel.random(len(models), d_embed, rng=rng)
    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(n_samples)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            batch_idx = order[start : start + batch_size]
            y_batch = targets[batch_idx]
            caches: list[dict[str, Array]] = []
            embeddings = []
            for model, x_site in zip(models, features):
                embedding, cache = model.forward(x_site[batch_idx], return_cache=True)
                embeddings.append(embedding)
                caches.append(cache)

            merged = np.concatenate(embeddings, axis=1)
            logits = server.predict_logits(merged)
            probabilities = _sigmoid(logits)
            bce = -np.mean(
                y_batch * np.log(probabilities + 1e-12)
                + (1.0 - y_batch) * np.log(1.0 - probabilities + 1e-12)
            )
            epoch_losses.append(float(bce))

            grad_logits = (probabilities - y_batch) / y_batch.shape[0]
            server_weights_before = server.weights.copy()
            server.weights -= lr_server * (merged.T @ grad_logits)
            server.bias -= lr_server * float(grad_logits.sum())

            for site_idx, model in enumerate(models):
                left = site_idx * d_embed
                right = left + d_embed
                grad_embedding = grad_logits[:, None] * server_weights_before[left:right][None, :]
                gradients = _backward_local(model, caches[site_idx], grad_embedding)
                _apply_local_gradients(model, gradients, lr_local)

        losses.append(float(np.mean(epoch_losses)))

    return Stage1Result(local_models=models, server_model=server, losses=losses)


def embedding_component_scores(embeddings: Array, y: Array) -> Array:
    """Score embedding dimensions by absolute point-biserial correlation."""

    values = _as_2d_float("embeddings", embeddings)
    targets = _as_binary_targets(y)
    if values.shape[0] != targets.shape[0]:
        raise ValueError("embeddings and y must have the same row count")

    centered_y = targets - targets.mean()
    centered_x = values - values.mean(axis=0, keepdims=True)
    numerator = centered_x.T @ centered_y
    denominator = np.sqrt((centered_x * centered_x).sum(axis=0) * np.sum(centered_y * centered_y))
    return np.abs(numerator / np.maximum(denominator, 1e-12))


def select_embedding_components(
    embeddings_by_site: Sequence[Array],
    y: Array,
    *,
    threshold: float = 0.05,
    top_k: int | None = None,
    min_components: int = 1,
) -> list[Array]:
    """Stage 2: server-side selection of useful embedding components."""

    selected: list[Array] = []
    for embeddings in embeddings_by_site:
        scores = embedding_component_scores(embeddings, y)
        if top_k is not None:
            count = min(max(top_k, min_components), scores.shape[0])
            chosen = np.argsort(scores)[-count:]
        else:
            chosen = np.flatnonzero(scores >= threshold)
            if chosen.shape[0] < min_components:
                chosen = np.argsort(scores)[-min_components:]
        selected.append(np.sort(chosen.astype(int)))
    return selected


def reconstruction_error(
    model: LocalTransVFLModel,
    teacher_model: LocalTransVFLModel,
    z: Array,
    selected_embedding_indices: Sequence[int] | Array,
) -> float:
    indices = np.asarray(selected_embedding_indices, dtype=int)
    if indices.size == 0:
        raise ValueError("selected_embedding_indices must not be empty")
    student = model.forward(z)[:, indices]
    teacher = teacher_model.forward(z)[:, indices]
    return float(np.mean((student - teacher) ** 2))


def local_feature_selection(
    z: Array,
    teacher_model: LocalTransVFLModel,
    selected_embedding_indices: Sequence[int] | Array,
    *,
    lambda_value: float,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.02,
    threshold: float = 1e-3,
    seed: int = 0,
) -> Stage3Result:
    """Stage 3: local distillation plus row-wise group-lasso pruning."""

    features = _as_2d_float("z", z)
    indices = np.asarray(selected_embedding_indices, dtype=int)
    if indices.size == 0:
        raise ValueError("selected_embedding_indices must not be empty")
    if np.any(indices < 0) or np.any(indices >= teacher_model.d_embed):
        raise ValueError("selected embedding index out of bounds")
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")

    rng = np.random.default_rng(seed)
    student = teacher_model.copy()
    n_samples = features.shape[0]
    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(n_samples)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            batch_idx = order[start : start + batch_size]
            x_batch = features[batch_idx]
            target = teacher_model.forward(x_batch)[:, indices]
            embedding, cache = student.forward(x_batch, return_cache=True)
            active_embedding = embedding[:, indices]
            diff = active_embedding - target
            mse = float(np.mean(diff * diff))
            epoch_losses.append(mse + lambda_value * group_lasso_penalty(student.w_select))

            grad_active = 2.0 * diff / diff.size
            grad_embedding = np.zeros_like(embedding)
            grad_embedding[:, indices] = grad_active
            gradients = _backward_local(student, cache, grad_embedding)

            student.w_select -= lr * gradients["w_select"]
            student.b_select -= lr * gradients["b_select"]
            student.w_select = proximal_group_lasso_rows(student.w_select, lr * lambda_value)

        losses.append(float(np.mean(epoch_losses)))

    kept, pruned = student.hard_prune(threshold)
    feature_norms = student.feature_norms()
    sparsity = float(pruned.shape[0] / feature_norms.shape[0])
    recon = reconstruction_error(student, teacher_model, features, indices)
    return Stage3Result(
        model=student,
        kept_features=kept,
        pruned_features=pruned,
        feature_norms=feature_norms,
        losses=losses,
        reconstruction_loss=recon,
        sparsity=sparsity,
        lambda_value=float(lambda_value),
        selected_embedding_indices=indices.copy(),
    )


def tune_lambda_grid(
    train_z: Array,
    validation_z: Array,
    teacher_model: LocalTransVFLModel,
    selected_embedding_indices: Sequence[int] | Array,
    lambdas: Sequence[float],
    *,
    gamma: float = 0.01,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 0.02,
    threshold: float = 1e-3,
    seed: int = 0,
) -> LambdaTuningResult:
    """Local lambda selection from Section 2.7 of the PDF."""

    lambda_values = [float(value) for value in lambdas]
    if not lambda_values:
        raise ValueError("lambdas must contain at least one value")

    scores: dict[float, float] = {}
    results: dict[float, Stage3Result] = {}
    for offset, lambda_value in enumerate(lambda_values):
        result = local_feature_selection(
            train_z,
            teacher_model,
            selected_embedding_indices,
            lambda_value=lambda_value,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            threshold=threshold,
            seed=seed + offset,
        )
        validation_loss = reconstruction_error(result.model, teacher_model, validation_z, selected_embedding_indices)
        score = validation_loss + gamma * (1.0 - result.sparsity)
        scores[lambda_value] = float(score)
        results[lambda_value] = result

    best_lambda = min(scores, key=scores.get)
    return LambdaTuningResult(
        best_lambda=best_lambda,
        best_result=results[best_lambda],
        scores=scores,
    )


def communication_cost(
    *,
    n_samples: int,
    d_embed: int,
    n_sites: int,
    standard_epochs: int,
    trans_stage1_epochs: int,
    stage2_sample_fraction: float = 0.1,
) -> dict[str, float]:
    """Return the embedding-count communication comparison from Section 2.5."""

    if not 0.0 < stage2_sample_fraction <= 1.0:
        raise ValueError("stage2_sample_fraction must be in (0, 1]")

    standard_vfl = standard_epochs * n_samples * d_embed * n_sites
    stage1 = trans_stage1_epochs * n_samples * d_embed * n_sites
    stage2 = int(np.ceil(stage2_sample_fraction * n_samples)) * d_embed * n_sites
    return {
        "standard_vfl": float(standard_vfl),
        "trans_vfl_stage1": float(stage1),
        "trans_vfl_stage2": float(stage2),
        "trans_vfl_total": float(stage1 + stage2),
    }


def feature_selection_metrics(
    selected_indices: Sequence[int] | Array,
    true_indices: Sequence[int] | Array,
    *,
    total_features: int,
) -> dict[str, float]:
    """Compute feature-selection metrics used in the simulations."""

    selected = set(np.asarray(selected_indices, dtype=int).tolist())
    truth = set(np.asarray(true_indices, dtype=int).tolist())
    if total_features <= 0:
        raise ValueError("total_features must be positive")

    true_positives = len(selected & truth)
    false_positives = len(selected - truth)
    true_negatives = total_features - len(truth) - false_positives
    false_negatives = len(truth - selected)

    return {
        "tp": float(true_positives),
        "fp": float(false_positives),
        "tn": float(true_negatives),
        "fn": float(false_negatives),
        "tpr": true_positives / max(len(truth), 1),
        "tnr": true_negatives / max(total_features - len(truth), 1),
        "fdr": false_positives / max(len(selected), 1),
        "selected": float(len(selected)),
    }
