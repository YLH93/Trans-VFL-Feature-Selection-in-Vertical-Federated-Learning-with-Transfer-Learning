# Trans-VFL

Communication-efficient learned feature selection for Vertical Federated Learning
with transfer-learning style frozen representations.

This repository implements the main workflow described in
`Project_3_Yilun (1).pdf`:

1. Each site keeps raw data local and uses a frozen base extractor to produce learned
   features.
2. A trainable local selection layer and embedding head produce the embeddings sent
   to the server.
3. The server selects statistically useful embedding components.
4. Each site disconnects from the server and locally prunes learned features with a
   group-lasso penalty.

The implementation is intentionally lightweight and uses NumPy so the core
algorithm and simulations can run without a deep-learning runtime. The notebooks
call the same package code as the CLI scripts.

## Repository layout

```text
trans_vfl/
  core.py          Core Trans-VFL model, Stage 1/2/3 routines, lambda tuning
  simulations.py   Synthetic EHR and genomic simulations from the paper
scripts/
  run_ehr_simulation.py
  run_genomic_simulation.py
tests/
  test_core.py
```

## Quick start

```powershell
python -m unittest discover
python .\scripts\run_ehr_simulation.py --patients 5000 --sites 20 --replications 25
python .\scripts\run_genomic_simulation.py --patients 200 --sites 10 --replications 100
```

If you install the package locally, the scripts can also be run from outside the
repository root:

```powershell
python -m pip install -e .
```

## Algorithm mapping

The code follows the notation in Section 2 of the PDF.

- `LocalTransVFLModel` implements the local architecture:
  frozen learned features `z_m`, trainable selection matrix
  `theta_select`, and a two-layer embedding head.
- `collaborative_pretrain` implements Stage 1 by training local selection/head
  parameters and a server fusion model with the base features frozen.
- `select_embedding_components` implements Stage 2 with a server-side
  correlation score against the binary outcome.
- `local_feature_selection` implements Stage 3 with a distillation loss on the
  selected embedding components plus row-wise group lasso on `theta_select`.
- `tune_lambda_grid` implements the local grid-search strategy from Section 2.7.

The selection-layer matrix is stored as `(d_base, d_select)`, so each row
corresponds to one learned feature from the frozen base. A row norm near zero
means that learned feature is pruned from all downstream computations.

## Simulations

`trans_vfl.simulations` includes two paper-aligned synthetic settings.

- EHR simulation: `M` sites, learned feature dimension `50`, with `20`
  predictive and `30` distractor features per site.
- Genomic simulation: pathway-level learned features derived from `5000`
  genes, with `25` predictive pathways and `225` noisy pathways.

The simulation runners average across repeated generated datasets and print the
observed Trans-VFL row next to the corresponding PDF target row. The PDF's table
metrics are not perfectly mutually consistent, so the implementation reports the
actual internally consistent FDR/TPR/TNR from each generated experiment while
calibrating the headline behavior around the reported accuracy and selected
feature counts.
