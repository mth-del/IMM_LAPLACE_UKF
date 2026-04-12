"""CSV 数据集融合（委托 run_fusion_dataset）。"""

from __future__ import annotations

from argparse import Namespace


def run_from_ns(ns: Namespace) -> None:
    from run_fusion_dataset import main as ds_main

    argv: list[str] = []
    if (d := (getattr(ns, "dataset_dir", "") or "").strip()):
        argv += ["--dataset-dir", d]
    if (o := (getattr(ns, "out_csv", "") or "").strip()):
        argv += ["--out-csv", o]
    if (p := (getattr(ns, "dataset_plots_dir", "") or "").strip()):
        argv += ["--plots-dir", p]
    if getattr(ns, "gui", False):
        argv.append("--gui")
    if getattr(ns, "no_plots", False):
        argv.append("--no-plots")
    if getattr(ns, "plot_only", False):
        argv.append("--plot-only")
    if hasattr(ns, "acc_noise_std"):
        argv += ["--acc-noise-std", str(ns.acc_noise_std)]
    if hasattr(ns, "dvl_noise_std"):
        argv += ["--dvl-noise-std", str(ns.dvl_noise_std)]
    if hasattr(ns, "usbl_dummy_std"):
        argv += ["--usbl-dummy-std", str(ns.usbl_dummy_std)]

    ds_main(argv)
