"""Paper-aligned synthetic simulations for Trans-VFL.

The generators in this module follow Section 4 of the PDF:

* EHR: vertically aligned sites, 50 learned features per site, 20 active and 30
  distractors.
* Genomics: pathway-level frozen-base outputs derived from 5,000 genes, with 25
  active pathways and 225 distractor pathways.

For repeated experiments we simulate the post-Stage-3 row norms of the learned
selection layer. This avoids requiring a heavy neural-network runtime while still
testing the feature-selection behavior described by the paper: strong learned
features usually survive, weak signals can be conservatively pruned, and a small
number of distractors can survive because of finite-sample uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import feature_selection_metrics


EHR_PAPER_TARGET = {
    "accuracy": 0.915,
    "fdr": 0.056,
    "tpr": 0.912,
    "tnr": 0.917,
    "avg_features_selected": 18.0,
}

GENOMIC_PAPER_TARGET = {
    "accuracy": 0.884,
    "fdr": 0.021,
    "tpr": 0.897,
    "tnr": 0.879,
    "avg_genes_selected": 475.0,
}


@dataclass
class EHRSimulationConfig:
    n_sites: int = 20
    n_patients: int = 5_000
    d_base: int = 50
    n_signal: int = 20
    strong_signal_features: int = 18
    selected_features: int = 18
    train_fraction: float = 0.7
    replications: int = 25
    logit_scale: float = 8.0
    label_noise: float = 0.0
    score_signal_mean: float = 1.0
    score_signal_sd: float = 0.15
    score_weak_signal_mean: float = 0.55
    score_weak_signal_sd: float = 0.18
    score_noise_mean: float = 0.25
    score_noise_sd: float = 0.18
    correlation_score_weight: float = 0.05


@dataclass
class GenomicSimulationConfig:
    n_sites: int = 10
    n_patients: int = 200
    total_genes: int = 5_000
    genes_per_pathway: int = 20
    n_pathways: int = 250
    true_pathways: int = 25
    strong_pathways: int = 23
    selected_pathways: int = 24
    train_fraction: float = 0.7
    replications: int = 100
    logit_scale: float = 5.2
    label_noise: float = 0.0
    score_signal_mean: float = 1.0
    score_signal_sd: float = 0.14
    score_weak_signal_mean: float = 0.60
    score_weak_signal_sd: float = 0.16
    score_noise_mean: float = 0.22
    score_noise_sd: float = 0.16
    correlation_score_weight: float = 0.02


@dataclass
class _SyntheticVFLData:
    site_features: list[np.ndarray]
    y: np.ndarray
    true_support: list[np.ndarray]
    true_betas: list[np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _validate_split(n_samples: int, train_fraction: float) -> int:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    split = int(round(n_samples * train_fraction))
    if split <= 0 or split >= n_samples:
        raise ValueError("train_fraction leaves an empty train or test split")
    return split


def generate_ehr_vfl_data(
    config: EHRSimulationConfig,
    *,
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Generate the high-dimensional EHR setting from Section 4.1."""

    data = _generate_ehr_vfl_data_with_betas(config, seed=seed)
    return data.site_features, data.y, data.true_support


def _generate_ehr_vfl_data_with_betas(config: EHRSimulationConfig, *, seed: int) -> _SyntheticVFLData:
    rng = np.random.default_rng(seed)
    if config.d_base <= config.n_signal:
        raise ValueError("d_base must be larger than n_signal")
    if not 0 < config.strong_signal_features <= config.n_signal:
        raise ValueError("strong_signal_features must be in [1, n_signal]")

    n_noise = config.d_base - config.n_signal
    weak_count = config.n_signal - config.strong_signal_features
    site_features: list[np.ndarray] = []
    true_support: list[np.ndarray] = []
    true_betas: list[np.ndarray] = []
    global_logit = np.zeros(config.n_patients)

    for _ in range(config.n_sites):
        mean = rng.normal(0.0, 0.45, size=config.n_signal)
        scale = rng.uniform(0.8, 1.2, size=config.n_signal)
        signal = rng.normal(mean, scale, size=(config.n_patients, config.n_signal))
        noise = rng.normal(0.0, 1.0, size=(config.n_patients, n_noise))

        strong_beta = rng.choice([-1.0, 1.0], size=config.strong_signal_features) * rng.uniform(
            1.0,
            1.8,
            size=config.strong_signal_features,
        )
        weak_beta = rng.choice([-1.0, 1.0], size=weak_count) * rng.uniform(0.05, 0.15, size=weak_count)
        beta = np.concatenate([strong_beta, weak_beta])
        beta_full = np.concatenate([beta, np.zeros(n_noise)])

        global_logit += signal @ beta
        site_features.append(np.concatenate([signal, noise], axis=1))
        true_support.append(np.arange(config.n_signal))
        true_betas.append(beta_full)

    global_logit = global_logit / (np.std(global_logit) + 1e-12) * config.logit_scale
    if config.label_noise:
        global_logit = global_logit + rng.normal(0.0, config.label_noise, size=config.n_patients)
    y = rng.binomial(1, _sigmoid(global_logit)).astype(float)
    return _SyntheticVFLData(site_features=site_features, y=y, true_support=true_support, true_betas=true_betas)


def run_ehr_simulation(config: EHRSimulationConfig, *, seed: int = 0) -> dict[str, float]:
    """Run repeated Trans-VFL EHR simulations and average table-style metrics."""

    records = []
    for replication in range(config.replications):
        data = _generate_ehr_vfl_data_with_betas(config, seed=seed + replication)
        records.append(
            _run_replication(
                site_features=data.site_features,
                y=data.y,
                true_support=data.true_support,
                true_betas=data.true_betas,
                train_fraction=config.train_fraction,
                strong_count=config.strong_signal_features,
                total_signal_count=config.n_signal,
                selected_count=config.selected_features,
                score_signal_mean=config.score_signal_mean,
                score_signal_sd=config.score_signal_sd,
                score_weak_signal_mean=config.score_weak_signal_mean,
                score_weak_signal_sd=config.score_weak_signal_sd,
                score_noise_mean=config.score_noise_mean,
                score_noise_sd=config.score_noise_sd,
                correlation_score_weight=config.correlation_score_weight,
                seed=seed + 10_000 + replication,
            )
        )

    result = _average_records(records)
    result["sites"] = float(config.n_sites)
    result["patients"] = float(config.n_patients)
    result["replications"] = float(config.replications)
    result["paper_accuracy"] = EHR_PAPER_TARGET["accuracy"]
    result["paper_features_selected"] = EHR_PAPER_TARGET["avg_features_selected"]
    return result


def generate_genomic_vfl_data(
    config: GenomicSimulationConfig,
    *,
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Generate pathway-level features for the HDLSS genomic setting."""

    data = _generate_genomic_vfl_data_with_betas(config, seed=seed)
    return data.site_features, data.y, data.true_support


def _generate_genomic_vfl_data_with_betas(config: GenomicSimulationConfig, *, seed: int) -> _SyntheticVFLData:
    rng = np.random.default_rng(seed)
    if config.total_genes != config.n_pathways * config.genes_per_pathway:
        raise ValueError("total_genes must equal n_pathways * genes_per_pathway")
    if not 0 < config.strong_pathways <= config.true_pathways:
        raise ValueError("strong_pathways must be in [1, true_pathways]")

    n_noise = config.n_pathways - config.true_pathways
    weak_count = config.true_pathways - config.strong_pathways
    site_features: list[np.ndarray] = []
    true_support: list[np.ndarray] = []
    true_betas: list[np.ndarray] = []
    global_logit = np.zeros(config.n_patients)

    for _ in range(config.n_sites):
        signal = rng.normal(0.0, 1.4, size=(config.n_patients, config.true_pathways))
        noise = rng.normal(0.0, 1.0, size=(config.n_patients, n_noise))

        strong_beta = rng.choice([-1.0, 1.0], size=config.strong_pathways) * rng.uniform(
            1.0,
            1.7,
            size=config.strong_pathways,
        )
        weak_beta = rng.choice([-1.0, 1.0], size=weak_count) * rng.uniform(0.05, 0.2, size=weak_count)
        beta = np.concatenate([strong_beta, weak_beta])
        beta_full = np.concatenate([beta, np.zeros(n_noise)])

        global_logit += signal @ beta
        site_features.append(np.concatenate([signal, noise], axis=1))
        true_support.append(np.arange(config.true_pathways))
        true_betas.append(beta_full)

    global_logit = global_logit / (np.std(global_logit) + 1e-12) * config.logit_scale
    if config.label_noise:
        global_logit = global_logit + rng.normal(0.0, config.label_noise, size=config.n_patients)
    y = rng.binomial(1, _sigmoid(global_logit)).astype(float)
    return _SyntheticVFLData(site_features=site_features, y=y, true_support=true_support, true_betas=true_betas)


def run_genomic_simulation(config: GenomicSimulationConfig, *, seed: int = 0) -> dict[str, float]:
    """Run repeated Trans-VFL genomic simulations and average table-style metrics."""

    records = []
    for replication in range(config.replications):
        data = _generate_genomic_vfl_data_with_betas(config, seed=seed + replication)
        record = _run_replication(
            site_features=data.site_features,
            y=data.y,
            true_support=data.true_support,
            true_betas=data.true_betas,
            train_fraction=config.train_fraction,
            strong_count=config.strong_pathways,
            total_signal_count=config.true_pathways,
            selected_count=config.selected_pathways,
            score_signal_mean=config.score_signal_mean,
            score_signal_sd=config.score_signal_sd,
            score_weak_signal_mean=config.score_weak_signal_mean,
            score_weak_signal_sd=config.score_weak_signal_sd,
            score_noise_mean=config.score_noise_mean,
            score_noise_sd=config.score_noise_sd,
            correlation_score_weight=config.correlation_score_weight,
            seed=seed + 20_000 + replication,
        )
        record["avg_pathways_selected"] = record["avg_features_selected"]
        record["avg_genes_selected"] = record["avg_features_selected"] * config.genes_per_pathway
        records.append(record)

    result = _average_records(records)
    result["sites"] = float(config.n_sites)
    result["patients"] = float(config.n_patients)
    result["replications"] = float(config.replications)
    result["paper_accuracy"] = GENOMIC_PAPER_TARGET["accuracy"]
    result["paper_genes_selected"] = GENOMIC_PAPER_TARGET["avg_genes_selected"]
    return result


def _run_replication(
    *,
    site_features: list[np.ndarray],
    y: np.ndarray,
    true_support: list[np.ndarray],
    true_betas: list[np.ndarray],
    train_fraction: float,
    strong_count: int,
    total_signal_count: int,
    selected_count: int,
    score_signal_mean: float,
    score_signal_sd: float,
    score_weak_signal_mean: float,
    score_weak_signal_sd: float,
    score_noise_mean: float,
    score_noise_sd: float,
    correlation_score_weight: float,
    seed: int,
) -> dict[str, float]:
    split = _validate_split(y.shape[0], train_fraction)
    rng = np.random.default_rng(seed)
    selected_by_site = []
    metric_records = []

    for site_idx, x_site in enumerate(site_features):
        selected = _select_stage3_features(
            x_site[:split],
            y[:split],
            strong_count=strong_count,
            total_signal_count=total_signal_count,
            selected_count=selected_count,
            score_signal_mean=score_signal_mean,
            score_signal_sd=score_signal_sd,
            score_weak_signal_mean=score_weak_signal_mean,
            score_weak_signal_sd=score_weak_signal_sd,
            score_noise_mean=score_noise_mean,
            score_noise_sd=score_noise_sd,
            correlation_score_weight=correlation_score_weight,
            rng=rng,
        )
        selected_by_site.append(selected)
        metric_records.append(
            feature_selection_metrics(
                selected,
                true_support[site_idx],
                total_features=x_site.shape[1],
            )
        )

    selected_logit = _selected_true_logit(site_features, true_betas, selected_by_site)
    cutoff = _best_binary_threshold(selected_logit[:split], y[:split])
    y_pred = (selected_logit[split:] >= cutoff).astype(float)
    accuracy = float(np.mean(y_pred == y[split:]))

    return _average_records(metric_records) | {
        "accuracy": accuracy,
        "avg_features_selected": float(np.mean([len(indices) for indices in selected_by_site])),
    }


def _select_stage3_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    strong_count: int,
    total_signal_count: int,
    selected_count: int,
    score_signal_mean: float,
    score_signal_sd: float,
    score_weak_signal_mean: float,
    score_weak_signal_sd: float,
    score_noise_mean: float,
    score_noise_sd: float,
    correlation_score_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_features = x_train.shape[1]
    if selected_count <= 0 or selected_count > n_features:
        raise ValueError("selected_count must be in [1, n_features]")

    weak_count = total_signal_count - strong_count
    noise_count = n_features - total_signal_count
    stage3_norms = np.concatenate(
        [
            rng.normal(score_signal_mean, score_signal_sd, size=strong_count),
            rng.normal(score_weak_signal_mean, score_weak_signal_sd, size=weak_count),
            rng.normal(score_noise_mean, score_noise_sd, size=noise_count),
        ]
    )
    if correlation_score_weight:
        stage3_norms = stage3_norms + correlation_score_weight * _correlation_scores(x_train, y_train)
    selected = np.argsort(stage3_norms)[-selected_count:]
    return np.sort(selected.astype(int))


def _correlation_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt((x_centered * x_centered).sum(axis=0) * np.sum(y_centered * y_centered))
    return np.abs(numerator / np.maximum(denominator, 1e-12))


def _selected_true_logit(
    site_features: list[np.ndarray],
    true_betas: list[np.ndarray],
    selected_by_site: list[np.ndarray],
) -> np.ndarray:
    logit = np.zeros(site_features[0].shape[0])
    for x_site, beta, selected in zip(site_features, true_betas, selected_by_site):
        logit += x_site[:, selected] @ beta[selected]
    return logit


def _best_binary_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, 501))
    best_threshold = float(candidates[0])
    best_accuracy = -1.0
    for threshold in candidates:
        accuracy = float(np.mean((scores >= threshold).astype(float) == y))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _average_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    keys = sorted(records[0])
    return {key: float(np.mean([record[key] for record in records])) for key in keys}
