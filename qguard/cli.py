from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sivc_marl.config import load_config
from sivc_marl.evaluation import evaluate_policy
from sivc_marl.model import SIVCMAPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the released SIVC-MAPPO actor")
    parser.add_argument("--weights", default="weights/sivc_mappo_actor.pt")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--output", default="outputs/evaluation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    policy = SIVCMAPPOPolicy.from_pretrained(args.weights, device=args.device)
    rows = evaluate_policy(policy, cfg, args.seeds, deterministic=not args.sample)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} evaluation episodes to {output}")
    for row in rows:
        print(
            f"seed={row['evaluation_seed']} "
            f"reward={row['total_reward']:.3f} "
            f"F1={row['confirmation_f1']:.3f} "
            f"latency={row['avg_service_latency_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
