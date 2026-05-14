"""Training script for oxide-supported metal cluster generative diffusion model.

Usage:
    # Command line:
    python -m oxide_diffusion.train --data data/AuNiCuPdPt-ZnO.arc --epochs 500

    # PyCharm / direct execution: just run this file.
"""

import argparse
import glob
import os
import sys

# Handle both `python -m oxide_diffusion.train` and direct execution from IDE
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, _proj_root)
    from oxide_diffusion.data.arc_parser import (
        auto_detect_oxide_type,
        frames_to_ase_atoms,
        parse_arc_file,
    )
    from oxide_diffusion.data.dataset import create_oxide_dataset
    from oxide_diffusion.model import get_oxide_diffusion_model
else:
    from .data.arc_parser import (
        auto_detect_oxide_type,
        frames_to_ase_atoms,
        parse_arc_file,
    )
    from .data.dataset import create_oxide_dataset
    from .model import get_oxide_diffusion_model

import pytorch_lightning as pl

from dss.utils.ema import EMA, EMACheckpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train oxide-supported metal cluster diffusion model"
    )
    # Data
    parser.add_argument(
        "--data", type=str, default=os.path.join(_proj_root, "data"),
        help="Path to .arc file, directory of .arc files, or glob pattern"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for checkpoints (default: ./output/<oxide_type>/)"
    )
    # Model
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--n_atom_basis", type=int, default=64)
    parser.add_argument("--n_rbf", type=int, default=30)
    parser.add_argument("--n_interactions", type=int, default=4)
    parser.add_argument("--gated_blocks", type=int, default=4)
    parser.add_argument("--beta_max", type=float, default=3.0)
    parser.add_argument("--beta_min", type=float, default=1e-2)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999,
        help="EMA decay rate"
    )

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ── Data ──────────────────────────────────────────────────────
    # Resolve to absolute path(s), support file / directory / glob
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(_proj_root, data_path)
    if os.path.isdir(data_path):
        arc_files = sorted(glob.glob(os.path.join(data_path, "*.arc")))
    elif "*" in data_path or "?" in data_path:
        arc_files = sorted(glob.glob(data_path))
    else:
        arc_files = [data_path]

    if not arc_files:
        print(f"No .arc files found at: {data_path}")
        sys.exit(1)

    print(f"Found {len(arc_files)} .arc file(s):")
    for f in arc_files:
        print(f"  {f}")

    # Parse and merge all files
    all_frames = []
    oxide_types = set()
    for f in arc_files:
        frames = parse_arc_file(f)
        ot = auto_detect_oxide_type(f)
        oxide_types.add(ot)
        print(f"  {os.path.basename(f)}: {len(frames)} frames, oxide={ot}")
        all_frames.extend(frames)
    frames = all_frames
    print(f"Total frames: {len(frames)}")

    if len(oxide_types) > 1:
        print(f"Warning: multiple oxide types detected: {oxide_types}")
        print("Using first oxide type for mask logic; verify this is correct.")
    oxide_type = sorted(oxide_types)[0]

    if args.output_dir is None:
        args.output_dir = os.path.join(_proj_root, "output", oxide_type)

    # Convert frames to ASE Atoms
    atoms_list = frames_to_ase_atoms(frames)
    print(f"Converted to {len(atoms_list)} ASE Atoms objects")

    # ── Model ─────────────────────────────────────────────────────
    diffusion, neighbour_list = get_oxide_diffusion_model(
        cutoff=args.cutoff,
        n_atom_basis=args.n_atom_basis,
        n_rbf=args.n_rbf,
        n_interactions=args.n_interactions,
        gated_blocks=args.gated_blocks,
        beta_max=args.beta_max,
        beta_min=args.beta_min,
        lr=args.lr,
    )

    # ── Dataset ───────────────────────────────────────────────────
    dataset_db = os.path.join(args.output_dir, "dataset.db")
    os.makedirs(args.output_dir, exist_ok=True)

    data_module = create_oxide_dataset(
        atoms_list=atoms_list,
        oxide_type=oxide_type,
        path=dataset_db,
        batch_size=args.batch_size,
        neighbour_list=neighbour_list,
        seed=args.seed,
    )

    # ── Trainer ───────────────────────────────────────────────────
    checkpoint_callback = EMACheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"{oxide_type}-{{epoch:04d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        every_n_epochs=50,
    )

    ema_callback = EMA(decay=args.ema_decay)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, ema_callback],
        log_every_n_steps=10,
        default_root_dir=args.output_dir,
    )

    print(f"Starting training for {args.epochs} epochs")
    print(f"  Oxide: {oxide_type}")
    print(f"  Model params: n_atom_basis={args.n_atom_basis}, "
          f"n_interactions={args.n_interactions}, cutoff={args.cutoff}")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Output: {args.output_dir}")

    trainer.fit(diffusion, datamodule=data_module)

    print(f"Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
