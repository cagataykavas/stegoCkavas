#!/usr/bin/env python3
"""
Sweep BPP values and summarize results in CSV/MD.

This script calls the existing pipeline as a subprocess. It uses sys.executable so it
runs inside your activated venv.

It does NOT require adding --force-generate/--force-prepare flags to *this* script;
those flags are forwarded to the pipeline with internal toggles (--force-generate/--force-prepare)
hard-coded here for reproducibility.

If your pipeline uses different flag names, change FORCE_FLAGS below.
"""
import argparse, os, csv, subprocess, sys, re

FORCE_FLAGS = ["--force-generate", "--force-prepare"]  # adjust if your pipeline uses different names

def parse_binary_macro_f1(stdout: str):
    # expects a line like: "Macro F1: 0.9360"
    m = re.search(r"Macro F1:\s*([0-9]*\.[0-9]+)", stdout)
    return float(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover-dir", required=True)
    ap.add_argument("--base-work-dir", required=True)
    ap.add_argument("--bpps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--algorithms", nargs="+", default=["lsb","pvd","dct","dwt"])
    ap.add_argument("--models", nargs="+", default=["rf","logreg","svm","xgb","lgb"])
    ap.add_argument("--feature-method", default="residual_cooc")
    ap.add_argument("--max-per-split", type=int, default=1000)
    ap.add_argument("--normalize-covers", action="store_true")
    ap.add_argument("--skip-multiclass", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.base_work_dir, exist_ok=True)

    rows = []
    for bpp in args.bpps:
        bpp_tag = str(bpp).replace(".", "p")
        work = os.path.join(args.base_work_dir, f"bpp_{bpp_tag}")

        cmd = [
            sys.executable, "-m", "stego_ai.pipeline",
            "--cover-dir", args.cover_dir,
            "--work-dir", work,
            "--bpp", str(bpp),
            "--algorithms", *args.algorithms,
            "--feature-method", args.feature_method,
            "--models", *args.models,
            "--save-models",
            "--max-per-split", str(args.max_per_split),
            *FORCE_FLAGS,
        ]
        if args.normalize_covers:
            cmd.append("--normalize-covers")
        if args.skip_multiclass:
            cmd.append("--skip-multiclass")

        print("\nRUN:", " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr)
            print(f"[fail] bpp={bpp} (exit={p.returncode}) -> skipping")
            continue

        # best-effort parse: only record rf macro f1 from stdout
        # (if you already save metrics.json, feel free to parse that instead)
        mf1 = parse_binary_macro_f1(p.stdout)
        rows.append({"bpp": bpp, "macro_f1_first_model": mf1})

    # write outputs
    out_csv = os.path.join(args.base_work_dir, "bpp_sweep_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["bpp","macro_f1_first_model"])
        w.writeheader()
        w.writerows(rows)

    out_md = os.path.join(args.base_work_dir, "bpp_sweep_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# BPP sweep summary\n\n")
        if not rows:
            f.write("_No successful runs._\n")
        else:
            f.write("| bpp | macro_f1_first_model |\n|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['bpp']} | {r['macro_f1_first_model']} |\n")

    print("Saved:", out_csv)
    print("Saved:", out_md)

if __name__ == "__main__":
    main()
