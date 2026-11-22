"""Shrink a training checkpoint to what prediction needs: the trained parameters only.

    cd OpenTSLM && export PYTHONPATH=$PWD:$PWD/src:/path/to/timenet-hackathon/src
    python .../slim_checkpoint.py --checkpoint .../best_model.pt --out .../model_slim.pt --verify

A ``best_model.pt`` holds the whole model (7.2 GB), the AdamW state (6.7 GB) and the scheduler.
``OpenTSLMFlamingo.load_from_file`` reads ``model_state`` alone and loads it with ``strict=False``
into a model whose frozen LLM layers it has just built from the base checkpoint on Hugging Face --
so only the parameters training changed need to be in the file (about 3.4 GB). That leaves room for
the 13 folds of PLAN.md 8.12.

``--verify`` proves it for this file: it loads the original and the slim checkpoint into two fresh
models and requires every tensor of the two to be identical. A trained parameter left out would keep
its random initialization in the second model and fail the check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TRAINED = ("vision_encoder", "perceiver", "gated_cross_attn", "embed_tokens", "lm_head")
"""Name parts of the parameters OpenTSLMFlamingo trains (its encoder, perceiver, cross-attention
layers and LM input embeddings, which Llama ties to the output layer); the rest is the frozen LLM."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a slim checkpoint already --verify'd: require the same parameter names, shapes and number of "
        "storages (seconds, instead of the minutes --verify takes; same code and model, so same tensors kept)",
    )
    parser.add_argument("--init-repo", default="OpenTSLM/llama-3.2-1b-ecg-flamingo", help="only to pick the base LLM")
    args = parser.parse_args()

    import torch

    full = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = full["model_state"]
    # No copies: each trained tensor is stored under up to four names (the ``model.`` and ``llm.``
    # aliases of one module, a cross-attention block reachable from two places, the embeddings tied
    # to the output layer). Kept as the same tensors, torch.save writes each storage once.
    kept = {k: v for k, v in state.items() if any(part in k for part in TRAINED)}
    slim = {"model_state": kept, "epoch": full.get("epoch"), "val_loss": full.get("val_loss"), "slim": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, args.out)
    del full, state
    print(f"kept {len(kept)} tensors; {args.checkpoint.stat().st_size / 1e9:.2f} GB -> {args.out.stat().st_size / 1e9:.2f} GB")

    if args.reference is not None:
        ref = torch.load(args.reference, map_location="cpu", mmap=True, weights_only=False)["model_state"]

        def layout(tensors: dict) -> tuple[dict, int]:
            shapes = {k: tuple(v.shape) for k, v in tensors.items()}
            return shapes, len({v.untyped_storage().data_ptr() for v in tensors.values()})

        if layout(kept) != layout(ref):
            raise SystemExit(f"FAIL: the kept tensors differ from the reference {args.reference}")
        print(f"MATCHES REFERENCE: same {len(kept)} names, shapes and storages as {args.reference}")
    if not args.verify:
        return
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_id = OpenTSLM._get_base_llm_id(args.init_repo)

    def loaded(path: Path):
        model = OpenTSLMFlamingo(device=device, llm_id=llm_id, cross_attn_every_n_layers=1, gradient_checkpointing=False)
        model.to(device).load_from_file(str(path))
        return model.state_dict()

    reference = loaded(args.checkpoint)
    candidate = loaded(args.out)
    if reference.keys() != candidate.keys():
        raise SystemExit(f"FAIL: the loaded models have different parameters: {set(reference) ^ set(candidate)}")
    different = [k for k in reference if not torch.equal(reference[k], candidate[k])]
    if different:
        raise SystemExit(f"FAIL: {len(different)} of {len(reference)} tensors differ, e.g. {different[:5]}")
    print(f"VERIFIED: all {len(reference)} tensors of the loaded model are identical")


if __name__ == "__main__":
    main()
