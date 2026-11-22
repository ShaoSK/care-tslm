"""Stage 6 of the OpenTSLM curriculum: wind-turbine early fault detection with rationale (CARE A+B).

Run from the root of an OpenTSLM checkout (it imports ``curriculum_learning.CurriculumTrainer``):

    cd OpenTSLM
    export PYTHONPATH=$PWD:$PWD/src:/path/to/timenet-hackathon/src
    # smoke test: tiny model, few samples, 1 epoch
    python /path/to/timenet-hackathon/src/train/stage6_wind_cot.py --store data/store_ab \
        --test-turbines A-0 --smoke --init-repo OpenTSLM/gemma-3-270m-ecg-flamingo
    # real fold
    python .../stage6_wind_cot.py --store data/store_ab --test-turbines A-0

Differences from stages 1-5 (everything else is CurriculumTrainer's own loop: AdamW, warmup,
early stopping on validation loss, best checkpoint in results/<llm>/<model>/<stage>/checkpoints/):

- the start weights come from a released OpenTSLM model on Hugging Face (default: the ECG-QA-CoT
  Flamingo, the last curriculum stage), not from a local stage-5 run;
- the stage's built-in test generation (batch 1, up to 2000 new tokens) is skipped. Hourly
  predictions for CARE-Score are made by ``predict.py``, batched and with a short token limit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.batching import SameFarmBatchSampler, farm_batch_sizes  # noqa: E402
from care_tslm.config import CHANNELS  # noqa: E402
from care_tslm.folds import NORMAL_PER_ANOMALY  # noqa: E402
from care_tslm.weighted_loss import DEFAULT_FIELD_WEIGHTS, make_weighted_compute_loss  # noqa: E402
from care_tslm.opentslm_dataset import FoldSpec, make_dataset_class  # noqa: E402

STAGE = "stage6_wind_cot"


def _curriculum_learning():
    """``curriculum_learning.py`` lives in the root of the OpenTSLM checkout, which is not on the
    path when this script is started by absolute path -- so fall back to the working directory."""
    try:
        import curriculum_learning
    except ModuleNotFoundError:
        sys.path.insert(0, os.getcwd())
        try:
            import curriculum_learning
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "curriculum_learning.py not found. Run this from the root of an OpenTSLM checkout, "
                "or add that checkout to PYTHONPATH."
            ) from exc
    return curriculum_learning


def make_trainer_class(field_weights: dict[str, float] | None = None):
    cl = _curriculum_learning()
    from huggingface_hub import hf_hub_download

    class WindCurriculumTrainer(cl.CurriculumTrainer):
        init_repo: str = "OpenTSLM/llama-3.2-1b-ecg-flamingo"

        def _initialize_model(self):
            """The upstream model, with the answer's fields weighted in the loss (PLAN.md 7).

            Without this the verdict -- the one token CARE-Score reads -- is about 1/40 of the
            gradient, the rest going into the explanation's wording. ``field_weights`` empty or all
            ones leaves the upstream loss untouched.
            """
            model = super()._initialize_model()
            if field_weights and any(w != 1 for w in field_weights.values()):
                model.compute_loss = make_weighted_compute_loss(model, field_weights)
                if self.rank == 0:
                    print(f"per-field loss weights: {field_weights}")
            return model

        def _load_previous_stage_model(self, current_stage):
            path = hf_hub_download(repo_id=self.init_repo, filename="model_checkpoint.pt")
            self._get_model().load_from_file(path)
            if self.rank == 0:
                print(f"Initialized {current_stage} from {self.init_repo}")
            return {"stage": self.init_repo, "metrics": {}, "epoch": "released", "val_loss": "n/a"}

        def _evaluate_stage(self, stage, test_loader, stage_name, metric_func=None, epoch=None):
            metrics = {"test_loss": None, "epoch": epoch, "note": "generation runs in predict.py"}
            out = Path(self.results_dir) / stage_name / "results"
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            return metrics

        def stage6_wind_cot(self, dataset_class, stage_name, *, epochs, lr, batch_size, eval_only=False):
            # One batch must stay within one farm: the farms have different sensor counts and the
            # Flamingo collate stacks a sample's curves into one tensor. The heavier farm also gets
            # a smaller batch, or it alone sets the memory ceiling (see care_tslm.batching).
            farms = set(dataset_class.train_farms)
            counts = dataset_class.fold_summary.get("channels_per_farm") or {
                farm: len(CHANNELS[farm]) for farm in farms
            }
            per_farm = farm_batch_sizes(batch_size, {farm: counts[farm] for farm in farms})
            if self.rank == 0:
                print(f"batch size per farm: {per_farm}")
            sampler = SameFarmBatchSampler(
                dataset_class.train_farms, batch_size, per_farm=per_farm, seed=dataset_class.fold.seed
            )
            return self._train_stage(
                stage_name=stage_name,
                dataset_class=dataset_class,
                num_epochs=epochs,
                lr_encoder=lr,
                lr_projector=lr / 2,
                lr_base=lr,
                metric_func=None,
                batch_size=batch_size,
                eval_only=eval_only,
                sampler=sampler,
            )

    return WindCurriculumTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--test-turbines", nargs="+", required=True, help="e.g. A-0 (leave-one-turbine-out)")
    parser.add_argument("--val-turbine", default="auto", help="'auto', a turbine such as A-11, or 'random'")
    parser.add_argument("--model", default="OpenTSLMFlamingo", choices=["OpenTSLMFlamingo", "OpenTSLMSP"])
    parser.add_argument("--llm-id", default=None, help="default: the base LLM of --init-repo")
    parser.add_argument("--init-repo", default="OpenTSLM/llama-3.2-1b-ecg-flamingo")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4, help="as stage 5 (ECG-QA-CoT)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--anomaly-stride-hours", type=int, default=1)
    parser.add_argument(
        "--ratio",
        type=int,
        default=NORMAL_PER_ANOMALY,
        help=f"normal:anomaly of the training set (default {NORMAL_PER_ANOMALY}); a non-default value "
        "gets its own stage name so it cannot overwrite another run's checkpoint",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the stage name, so a run with different settings cannot overwrite another "
        "run's checkpoint (the epoch count changes the LR schedule, so it changes the weights)",
    )
    parser.add_argument(
        "--field-weights",
        default=",".join(f"{k}={v:g}" for k, v in DEFAULT_FIELD_WEIGHTS.items()),
        help="per-field loss weights as 'Answer:=10,Component:=3'; 'none' keeps the upstream loss",
    )
    parser.add_argument("--smoke", action="store_true", help="64 train / 16 val samples, 1 epoch")
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="train/predict with 'Answer: anomaly|normal' alone, no Component or Evidence field",
    )
    parser.add_argument(
        "--answer-format",
        choices=("fields", "prefix", "last"),
        default="fields",
        help="answer field order: verdict first (fields), a fixed assessment sentence before the verdict "
        "(prefix), or verdict after the evidence (last); see care_tslm.text.FIELD_ORDER",
    )
    parser.add_argument(
        "--legacy-text",
        action="store_true",
        help="train/predict with the 11-18 prompt and prose answer (verdict last) -- the v1 "
        "configuration that scored CARE 0.7345; see text.PRE_PROMPT_LEGACY",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="anomaly samples only from config.EVIDENCE_EVENTS (PLAN.md 8.13); the test split is unchanged",
    )
    parser.add_argument(
        "--offset-aug",
        type=float,
        default=0.0,
        help="training only: add a random constant in +-this many sigma to every component residual",
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    spec = FoldSpec(
        store=args.store.resolve(),
        test_turbines=tuple(args.test_turbines),
        answer_only=args.answer_only,
        legacy_text=args.legacy_text,
        answer_format=args.answer_format,
        val_turbine=None if args.val_turbine == "random" else args.val_turbine,
        anomaly_stride_hours=args.anomaly_stride_hours,
        ratio=args.ratio,
        evidence_only=args.evidence_only,
        offset_aug=args.offset_aug,
        max_train=64 if args.smoke else None,
        max_val=16 if args.smoke else 400,
        max_test=16 if args.smoke else None,
    )
    if args.llm_id is None:
        from opentslm.model.llm.OpenTSLM import OpenTSLM

        args.llm_id = OpenTSLM._get_base_llm_id(args.init_repo)
    dataset_class = make_dataset_class(spec)
    print(json.dumps(dataset_class.fold_summary, indent=2))

    stage_name = (
        f"{STAGE}_{spec.name}"
        + ("_answeronly" if args.answer_only else "")
        + ("_legacy" if args.legacy_text else "")
        + (f"_{args.answer_format}" if args.answer_format != "fields" else "")
        + (f"_ratio{args.ratio}" if args.ratio != NORMAL_PER_ANOMALY else "")
        + ("_evid" if args.evidence_only else "")
        + (f"_aug{args.offset_aug:g}" if args.offset_aug > 0 else "")
        + (f"_{args.tag}" if args.tag else "")
        + ("_smoke" if args.smoke else "")
    )
    weights: dict[str, float] = {}
    if args.field_weights.lower() not in ("none", ""):
        for item in args.field_weights.split(","):
            marker, _, value = item.partition("=")
            weights[marker.strip()] = float(value)
    trainer_class = make_trainer_class(weights)
    trainer_class.init_repo = args.init_repo
    trainer = trainer_class(
        args.model,
        gradient_checkpointing=args.gradient_checkpointing,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        llm_id=args.llm_id,
    )
    summary_dir = Path(trainer.results_dir) / stage_name / "results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "fold_summary.json").write_text(json.dumps(dataset_class.fold_summary, indent=2))
    trainer.stage6_wind_cot(
        dataset_class,
        stage_name,
        epochs=1 if args.smoke else args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
