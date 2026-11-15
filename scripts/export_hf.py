"""
Export a pretrained checkpoint as a weights-only release file for the Hugging Face hub.

The file is a ``torch.save`` dict that ``probe.py --ckpt`` loads like a training checkpoint
(see ``models.load_pretrained``):
- ``model_state_dict``: ``backbone.*`` weights of ``ImageSSL``, exactly as trained;
- ``epoch``: last pretraining epoch (0-indexed, as in training checkpoints);
- ``config``: the training config, without cluster/logging keys, with the loss type in this
  repository's naming (``peira_comp`` of the paper runs is ``peira``);
- ``metadata``: method, projector dims, seed, epochs and offline linear-probe top-1.

The projector, optimizer, scheduler, AMP scaler, loss buffers and online probe head are
dropped: the release is for linear probing and transfer of the backbone features.

Usage:
    python -m scripts.export_hf --ckpt <run_dir>/latest.pth.tar --out-dir hf_staging/
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.transforms import get_dataset_info
from eb_jepa.utils.logging import get_logger
from models import TorchvisionResNet, load_pretrained

logger = get_logger(__name__)

# The renames, dropped keys and the "batch" fallback of release_config only convert the configs
# of the original paper runs, which predate this repository; its own configs need none of them.
LOSS_TYPE_RENAMES = {"peira_comp": "peira"}
LOSS_TYPE_TO_METHOD = {
    "peira": "peira",
    "bcs": "sigreg",
    "vicreg": "vicreg",
    "vicreg_ref": "vicreg",
}
DATASET_TAGS = {"imagenet1k": "in1k", "cifar10": "c10"}
ARCH_TAGS = {"resnet50": "rn50", "resnet18": "rn18"}
DROPPED_KEYS = ("slurm", "logging", "sweep", "sweep_name", "force_resume")


def release_config(run_dir: Path) -> dict:
    """``config.yaml`` of ``run_dir`` without cluster, logging and path keys, loss type renamed."""
    cfg = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=False)
    for key in DROPPED_KEYS:
        cfg.pop(key, None)
    cfg["meta"] = {"seed": cfg["meta"]["seed"]}
    cfg["data"].pop("data_dir", None)
    loss = {k: v for k, v in cfg["loss"].items() if v is not None}
    loss["type"] = LOSS_TYPE_RENAMES.get(loss["type"], loss["type"])
    if loss["type"] == "peira":
        loss.setdefault("predictor_cov", "batch")
    cfg["loss"] = loss
    return cfg


def release_metadata(cfg: dict, epoch: int, run_dir: Path) -> dict:
    """Method, dims, seed, epochs and offline top-1 (``offline_probe.json`` of ``run_dir``, if any)."""
    probe_path = run_dir / "offline_probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    if probe is None:
        logger.warning(f"no {probe_path}: run probe.py first to record the offline top-1")
    elif probe["pretrain_epoch"] != epoch:
        raise ValueError(f"offline_probe.json probed epoch {probe['pretrain_epoch']}, not {epoch}")
    return {
        "method": LOSS_TYPE_TO_METHOD[cfg["loss"]["type"]],
        "dataset": cfg["data"]["dataset"],
        "arch": cfg["model"]["resnet_arch"],
        "proj_hidden_dim": cfg["model"]["proj_hidden_dim"],
        "proj_output_dim": cfg["model"]["proj_output_dim"],
        "seed": cfg["meta"]["seed"],
        "epochs": epoch + 1,
        "offline_probe_top1": probe["final_top1"] if probe else None,
        "offline_probe_top5": probe["final_top5"] if probe else None,
    }


def release_name(meta: dict) -> str:
    """E.g. ``in1k_rn50_sigreg_seed1.pth``; PEIRA files carry k: ``in1k_rn50_peira_k256_seed1``."""
    method = meta["method"]
    if method == "peira":
        method += f"_k{meta['proj_output_dim']}"
    return (
        f"{DATASET_TAGS[meta['dataset']]}_{ARCH_TAGS[meta['arch']]}_{method}_seed{meta['seed']}.pth"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="train.py checkpoint")
    parser.add_argument(
        "--run-dir", type=str, default=None, help="holds config.yaml, offline_probe.json"
    )
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.ckpt).parent

    sd, ckpt = load_pretrained(args.ckpt)
    sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    cfg = release_config(run_dir)
    meta = release_metadata(cfg, ckpt["epoch"], run_dir)
    image_size, _ = get_dataset_info(meta["dataset"])
    backbone = TorchvisionResNet(arch=meta["arch"], image_size=image_size)
    backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in sd.items()})

    out = Path(args.out_dir) / release_name(meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": sd, "epoch": ckpt["epoch"], "config": cfg, "metadata": meta}, out
    )
    exported = torch.load(out, map_location="cpu", weights_only=True)
    assert exported["model_state_dict"].keys() == sd.keys()
    for k, v in sd.items():
        w = exported["model_state_dict"][k]
        assert w.dtype == v.dtype and w.shape == v.shape and torch.equal(w, v), k
    print(f"{out} {out.stat().st_size / 2**20:.1f} MiB, {len(sd)} tensors, {meta}")


if __name__ == "__main__":
    main()
