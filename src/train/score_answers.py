"""Score both verdicts of every test snapshot by likelihood, instead of generating one (PLAN.md 8.10).

    cd OpenTSLM && export PYTHONPATH=$PWD:$PWD/src:/path/to/timenet-hackathon/src
    python .../score_answers.py --store ~/data/store_ab_v3 --test-turbines A-10 --answer-only \
        --checkpoint results/.../stage6_wind_cot_test_A-0_answeronly_ao/checkpoints/best_model.pt \
        --out predictions/scored_A-10_ao.jsonl

Greedy decoding says ``anomaly`` only when the model puts more than half its mass on it, and a model
trained on 80 % normal rarely does. Here each snapshot is fed twice, once with each candidate answer
of ``text.verdict_candidates`` (the answer up to and including the verdict), through the very path
training uses (``pad_and_apply_batch`` -> ``model``), and the summed log-probability of each
candidate's tokens gives P(anomaly). ``src/scripts/tune_threshold.py`` then picks a threshold on the
fold's validation turbine.

Writes one JSON line per snapshot: the ``predict.py`` keys plus ``logp_anomaly``, ``logp_normal``,
``p_anomaly`` and ``pred`` (the argmax, i.e. what greedy decoding of the verdict would say).
Already written snapshots are skipped, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import text  # noqa: E402
from care_tslm.opentslm_dataset import FoldSpec, make_dataset_class  # noqa: E402

KEYS = ("farm", "event_id", "turbine", "end_step", "end_us", "status_end", "label")
LABELS = ("anomaly", "normal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--test-turbines", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-repo", default="OpenTSLM/llama-3.2-1b-ecg-flamingo", help="only to pick the base LLM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="snapshots per forward pass (x2 sequences)")
    parser.add_argument("--test-stride-hours", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--answer-format", choices=("fields", "prefix", "last"), default="fields")
    parser.add_argument(
        "--prefix-from",
        type=Path,
        default=None,
        help="predict.py output of the same checkpoint: each candidate becomes the model's own answer up "
        "to its 'Answer:' plus the verdict, i.e. P(anomaly | its own evidence). Needed for --answer-format last",
    )
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="score the test turbines' training-period snapshots instead (healthy, never trained on), "
        "to set each turbine's own threshold: scripts/calibrate_threshold.py",
    )
    parser.add_argument("--calibration-every", type=int, default=1, help="keep every n-th calibration snapshot")
    args = parser.parse_args()

    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    if args.answer_format == "last" and args.prefix_from is None:
        parser.error("--answer-format last needs --prefix-from: its verdict follows text the model writes")
    if args.prefix_from is None:
        fixed = text.verdict_candidates(args.answer_format, answer_only=args.answer_only)
    else:
        fixed = None
        with args.prefix_from.open() as f:
            generated = {(r["farm"], r["event_id"], r["end_step"]): r["generated"] for r in map(json.loads, f)}
    spec = FoldSpec(
        store=args.store.resolve(),
        test_turbines=tuple(args.test_turbines),
        answer_only=args.answer_only,
        answer_format=args.answer_format,
        test_stride_hours=args.test_stride_hours,
        calibration=args.calibration,
        calibration_every=args.calibration_every,
        max_test=args.limit,
        max_val=1,
        max_train=1,  # only the test split is used here
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpenTSLMFlamingo(
        device=device,
        llm_id=OpenTSLM._get_base_llm_id(args.init_repo),
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
    ).to(device)
    model.load_from_file(str(args.checkpoint))
    model.eval()
    # Right padding, as in training: pad_and_apply_batch marks the answer as everything after the
    # prompt's token count, which only holds when the prompt starts at index 0. (predict.py switches
    # to left padding for generation; scoring is not generation.)
    model.text_tokenizer.padding_side = "right"
    test = make_dataset_class(spec)("test", EOS_TOKEN=model.get_eos_token())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {(r["farm"], r["event_id"], r["end_step"]) for r in map(json.loads, f)}
    todo = [s for s in test if (s["farm"], s["event_id"], s["end_step"]) not in done]
    print(f"{len(test)} test snapshots, {len(done)} already done, {len(todo)} to go")
    print(f"candidates: {fixed}" if fixed else f"candidates: own answer up to 'Answer:' from {args.prefix_from}")

    def candidates_for(sample: dict) -> dict[str, str]:
        if fixed is not None:
            return fixed
        own = text.before_verdict(generated[(sample["farm"], sample["event_id"], sample["end_step"])]) or ""
        return {label: f"{own}Answer: {label}" for label in LABELS}

    started = time.time()
    with args.out.open("a") as out, torch.inference_mode():
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i : i + args.batch_size]
            # Each snapshot twice: [s0+anomaly, s0+normal, s1+anomaly, ...].
            pairs = [dict(s) | {"answer": candidates_for(s)[label]} for s in chunk for label in LABELS]
            batch = extend_time_series_to_match_patch_size_and_aggregate(pairs, patch_size=PATCH_SIZE)
            input_ids, images, attention_mask, labels = model.pad_and_apply_batch(batch, include_labels=False)
            output = model.model(vision_x=images, lang_x=input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits if hasattr(output, "logits") else output[1]
            # Position t predicts token t + 1; sum log-probs over each sequence's answer tokens.
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100
            picked = logits[:, :-1, :][mask].float()
            token_logp = picked.log_softmax(-1).gather(1, shift_labels[mask].unsqueeze(1)).squeeze(1)
            seq = mask.nonzero()[:, 0]
            logp = torch.zeros(len(pairs), device=token_logp.device).index_add_(0, seq, token_logp).tolist()
            for j, sample in enumerate(chunk):
                la, ln = logp[2 * j], logp[2 * j + 1]
                p_anomaly = 1.0 / (1.0 + math.exp(ln - la))
                row = {k: sample[k] for k in KEYS} | {
                    "logp_anomaly": la,
                    "logp_normal": ln,
                    "p_anomaly": p_anomaly,
                    "pred": "anomaly" if la > ln else "normal",
                }
                out.write(json.dumps(row) + "\n")
            out.flush()
            n = i + len(chunk)
            rate = n / (time.time() - started)
            print(f"{n}/{len(todo)}  {rate:.2f} samples/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
