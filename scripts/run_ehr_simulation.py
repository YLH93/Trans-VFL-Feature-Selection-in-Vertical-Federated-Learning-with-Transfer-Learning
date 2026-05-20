from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trans_vfl.simulations import EHR_PAPER_TARGET, EHRSimulationConfig, run_ehr_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Trans-VFL synthetic EHR simulation.")
    parser.add_argument("--patients", type=int, default=5000)
    parser.add_argument("--sites", type=int, default=20)
    parser.add_argument("--replications", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = EHRSimulationConfig(
        n_sites=args.sites,
        n_patients=args.patients,
        replications=args.replications,
    )
    result = run_ehr_simulation(config, seed=args.seed)
    print("Trans-VFL EHR simulation")
    print(f"replications: {int(result['replications'])}")
    print(f"sites: {int(result['sites'])}")
    print(f"patients: {int(result['patients'])}")
    print()
    print("metric                  observed    pdf_target")
    print(f"accuracy                {result['accuracy']:.4f}      {EHR_PAPER_TARGET['accuracy']:.4f}")
    print(f"fdr                     {result['fdr']:.4f}      {EHR_PAPER_TARGET['fdr']:.4f}")
    print(f"tpr                     {result['tpr']:.4f}      {EHR_PAPER_TARGET['tpr']:.4f}")
    print(f"tnr                     {result['tnr']:.4f}      {EHR_PAPER_TARGET['tnr']:.4f}")
    print(
        "avg_features_selected   "
        f"{result['avg_features_selected']:.2f}       {EHR_PAPER_TARGET['avg_features_selected']:.2f}"
    )


if __name__ == "__main__":
    main()
