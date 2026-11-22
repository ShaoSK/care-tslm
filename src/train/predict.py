"""Generate rationales + labels for every test snapshot of a fold (hourly, for CARE-Score).

    cd OpenTSLM && export PYTHONPATH=$PWD:$PWD/src:/path/to/timenet-hackathon/src
    python .../predict.py --store data/store_ab --test-turbines A-0 \
        --checkpoint results/Llama_3_2_1B/OpenTSLMFlamingo/stage6_wind_cot_test_A-0/checkpoints/best_model.pt \
        --out predictions/test_A-0.jsonl

Writes one JSON line per snapshot: farm, event_id, end_step, end_us, label (gold), pred, generated.
Already written snapshots are skipped, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.opentslm_dataset import FoldSpec, make_dataset_class, parse_answer  # noqa: E402

KEYS = ("farm", "event_id", "turbine", "end_step", "end_us", "status_end", "label")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--test-turbines", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-repo", default="OpenTSLM/llama-3.2-1b-ecg-flamingo", help="only to pick the base LLM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320, help="our rationales are ~80-150 tokens")
    parser.add_argument("--test-stride-hours", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
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
        "--answer-only",
        action="store_true",
        help="train/predict with 'Answer: anomaly|normal' alone, no Component or Evidence field",
    )
    args = parser.parse_args()

    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    spec = FoldSpec(
        store=args.store.resolve(),
        test_turbines=tuple(args.test_turbines),
        answer_only=args.answer_only,
        legacy_text=args.legacy_text,
        answer_format=args.answer_format,
        test_stride_hours=args.test_stride_hours,
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
    # Batched generation with a decoder-only LLM needs left padding: with OpenTSLM's default right
    # padding the shorter prompts in a batch continue *after* their pad tokens and decode to
    # nonsense. Only the generation path is affected -- training builds its labels assuming the
    # prompt starts at index 0, so this is set here and not in the trainer.
    model.text_tokenizer.padding_side = "left"
    test = make_dataset_class(spec)("test", EOS_TOKEN=model.get_eos_token())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {(r["farm"], r["event_id"], r["end_step"]) for r in map(json.loads, f)}
    todo = [s for s in test if (s["farm"], s["event_id"], s["end_step"]) not in done]
    print(f"{len(test)} test snapshots, {len(done)} already done, {len(todo)} to go")

    started = time.time()
    with args.out.open("a") as out:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i : i + args.batch_size]
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                [dict(s) for s in chunk], patch_size=PATCH_SIZE
            )
            generated = model.generate(batch, max_new_tokens=args.max_new_tokens)
            for sample, text in zip(chunk, generated, strict=True):
                row = {k: sample[k] for k in KEYS} | {"pred": parse_answer(text), "generated": text.strip()}
                out.write(json.dumps(row) + "\n")
            out.flush()
            n = i + len(chunk)
            rate = n / (time.time() - started)
            print(f"{n}/{len(todo)}  {rate:.2f} samples/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
