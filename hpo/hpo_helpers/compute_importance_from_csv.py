"""
Compute tunable importance and generate all Optuna plots from a CSV experiment output file.

Usage:
    python3 compute_importance_from_csv.py \
        --csv "/path/to/experiment-output (14).csv" \
        --json /path/to/quarkus_test1.json \
        [--output-dir ./output]

The CSV must have:
  - a column named 'Trial'
  - a column named 'objfn_result'
  - one column per tunable matching the names in the search space JSON

Rows with empty or negative objfn_result are treated as pruned/failed and excluded.

Output (in --output-dir):
  - importance_report.html       : Combined importance report with bar charts
  - tunable_importance.html
  - optimization_history.html
  - slice.html
  - parallel_coordinate.html
  - contour.html
  - rank.html
  - timeline.html
  - edf.html
"""

import argparse
import csv
import json
import os
import sys

import optuna
from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_search_space(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def load_csv(csv_path: str) -> list:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study(rows: list, tunables: list, direction: str) -> optuna.Study:
    """Reconstruct an in-memory Optuna study from CSV rows."""
    study = optuna.create_study(direction=direction)

    for row in rows:
        raw_result = row.get("objfn_result", "").strip()
        if not raw_result:
            continue
        try:
            obj_value = float(raw_result)
        except ValueError:
            continue
        if obj_value < 0:
            continue

        params = {}
        distributions = {}
        skip_row = False

        for t in tunables:
            name = t["name"]
            raw = row.get(name, "").strip()
            if not raw:
                skip_row = True
                break
            vtype = t["value_type"].lower()
            try:
                if vtype == "integer":
                    params[name] = int(raw)
                    distributions[name] = optuna.distributions.IntDistribution(
                        low=int(t["lower_bound"]),
                        high=int(t["upper_bound"]),
                        step=int(t.get("step", 1)),
                    )
                elif vtype == "double":
                    params[name] = float(raw)
                    distributions[name] = optuna.distributions.FloatDistribution(
                        low=float(t["lower_bound"]),
                        high=float(t["upper_bound"]),
                        step=float(t["step"]) if t.get("step") else None,
                    )
                elif vtype == "categorical":
                    params[name] = str(raw)
                    distributions[name] = optuna.distributions.CategoricalDistribution(
                        choices=t["choices"]
                    )
            except (ValueError, KeyError):
                skip_row = True
                break

        if skip_row:
            continue

        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=obj_value,
        )
        study.add_trial(trial)

    return study


# ---------------------------------------------------------------------------
# Importance computation
# ---------------------------------------------------------------------------

def compute_importance(study: optuna.Study) -> tuple:
    """Returns (fanова_dict, mdi_dict) — either may be None on failure."""
    fanova_result = None
    mdi_result = None

    try:
        fanова_result = optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator()
        )
    except Exception as e:
        print(f"  fANOVA failed: {e}")

    try:
        mdi_result = optuna.importance.get_param_importances(
            study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
        )
    except Exception as e:
        print(f"  MeanDecreaseImpurity failed: {e}")

    return fanова_result, mdi_result


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

PLOTS = [
    ("tunable_importance",   "plot_param_importances"),
    ("optimization_history", "plot_optimization_history"),
    ("slice",                "plot_slice"),
    ("parallel_coordinate",  "plot_parallel_coordinate"),
    ("contour",              "plot_contour"),
    ("rank",                 "plot_rank"),
    ("timeline",             "plot_timeline"),
    ("edf",                  "plot_edf"),
]


def generate_plots(study: optuna.Study, output_dir: str, experiment_name: str) -> dict:
    """Generate all Optuna plots. Returns dict of plot_type -> filepath or error message."""
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for plot_type, fn_name in PLOTS:
        filepath = os.path.join(output_dir, plot_type + ".html")
        try:
            plot_fn = getattr(optuna.visualization, fn_name)
            fig = plot_fn(study)
            fig.write_html(filepath)
            print(f"  [OK] {plot_type}.html")
            results[plot_type] = filepath
        except Exception as e:
            msg = f"Could not generate {plot_type}: {e}"
            print(f"  [SKIP] {msg}")
            # Write a placeholder so the file exists
            with open(filepath, "w") as f:
                f.write(f"<html><body><h2>{msg}</h2></body></html>")
            results[plot_type] = filepath

    return results


# ---------------------------------------------------------------------------
# HTML importance report
# ---------------------------------------------------------------------------

def _bar(score: float, color: str) -> str:
    pct = round(score * 100, 2)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<div style="width:180px;font-size:13px;color:#1f2328;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis" title="{{}}">{{}}</div>'
        f'<div style="flex:1;background:#e5e7eb;border-radius:3px;height:16px">'
        f'<div style="width:{pct}%;background:{color};height:16px;border-radius:3px"></div></div>'
        f'<div style="width:48px;text-align:right;font-size:13px;color:#57606a">{pct:.1f}%</div>'
        f'</div>'
    )


def build_importance_html(
    experiment_name: str,
    direction: str,
    total_rows: int,
    complete_count: int,
    skipped_count: int,
    fanova: dict,
    mdi: dict,
    plot_files: dict,
    output_dir: str,
) -> str:

    def section(title: str, importance: dict, color: str) -> str:
        if not importance:
            return f'<div class="card"><h3>{title}</h3><p class="muted">Not available.</p></div>'
        rows_html = ""
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            pct = round(score * 100, 2)
            rows_html += (
                f'<div class="row">'
                f'<div class="label" title="{param}">{param}</div>'
                f'<div class="bar-wrap"><div class="bar" style="width:{pct}%;background:{color}"></div></div>'
                f'<div class="score">{pct:.1f}%</div>'
                f'</div>\n'
            )
        return f'<div class="card"><h3>{title}</h3>{rows_html}</div>'

    plots_html = ""
    for plot_type, filepath in plot_files.items():
        label = plot_type.replace("_", " ").title()
        rel = os.path.basename(filepath)
        plots_html += f'<a href="{rel}" class="plot-link">{label}</a>\n'

    fanova_section = section("Tunable Importance — fANOVA", fanova, "#3b82d4")
    mdi_section = section("Tunable Importance — Mean Decrease Impurity", mdi, "#7c5cd8")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tunable Importance — {experiment_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Segoe UI", system-ui, sans-serif; font-size: 14px;
          line-height: 1.6; background: #ffffff; color: #1f2328; padding: 32px 16px; }}
  .wrap {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #57606a; font-size: 13px; margin-bottom: 24px; }}
  .meta {{ display: flex; gap: 24px; flex-wrap: wrap; background: #f7f8fa;
           border: 1px solid #e5e7eb; border-radius: 6px; padding: 14px 18px;
           margin-bottom: 24px; }}
  .meta-item {{ display: flex; flex-direction: column; gap: 2px; }}
  .meta-item .key {{ font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #57606a; }}
  .meta-item .val {{ font-size: 14px; font-weight: 500; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 20px;
           background: #ffffff; margin-bottom: 20px; }}
  .card h3 {{ font-size: 15px; font-weight: 600; margin-bottom: 14px;
              padding-bottom: 8px; border-bottom: 1px solid #e5e7eb; }}
  .row {{ display: flex; align-items: center; gap: 10px; margin: 5px 0; }}
  .label {{ width: 280px; font-size: 13px; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; color: #1f2328; }}
  .bar-wrap {{ flex: 1; background: #e5e7eb; border-radius: 3px; height: 14px; }}
  .bar {{ height: 14px; border-radius: 3px; min-width: 2px; }}
  .score {{ width: 48px; text-align: right; font-size: 13px; color: #57606a; }}
  .muted {{ color: #57606a; font-size: 13px; }}
  .plots-card {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 20px;
                 background: #ffffff; margin-bottom: 20px; }}
  .plots-card h3 {{ font-size: 15px; font-weight: 600; margin-bottom: 14px;
                    padding-bottom: 8px; border-bottom: 1px solid #e5e7eb; }}
  .plot-link {{ display: inline-block; margin: 4px 6px 4px 0;
                padding: 5px 12px; background: #f7f8fa; border: 1px solid #e5e7eb;
                border-radius: 4px; color: #3b82d4; font-size: 13px;
                text-decoration: none; }}
  footer {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #e5e7eb;
            text-align: center; font-size: 12px; color: #57606a; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Tunable Importance Report</h1>
  <div class="subtitle">{experiment_name}</div>

  <div class="meta">
    <div class="meta-item"><span class="key">Direction</span><span class="val">{direction}</span></div>
    <div class="meta-item"><span class="key">CSV Rows</span><span class="val">{total_rows}</span></div>
    <div class="meta-item"><span class="key">Complete Trials</span><span class="val">{complete_count}</span></div>
    <div class="meta-item"><span class="key">Skipped (failed/empty)</span><span class="val">{skipped_count}</span></div>
  </div>

  {fanova_section}
  {mdi_section}

  <div class="plots-card">
    <h3>Generated Plots</h3>
    {plots_html}
  </div>

  <footer>Made with IBM Bob</footer>
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute tunable importance and generate Optuna plots from CSV")
    parser.add_argument("--csv",        required=True, help="Path to experiment-output CSV file")
    parser.add_argument("--json",       required=True, help="Path to search space JSON file")
    parser.add_argument("--output-dir", default="./importance_output", help="Directory to write output files (default: ./importance_output)")
    args = parser.parse_args()

    search_space = load_search_space(args.json)
    tunables     = search_space["tunables"]
    direction    = search_space.get("direction", "maximize")
    exp_name     = search_space["experiment_name"]

    print(f"Experiment : {exp_name}")
    print(f"Direction  : {direction}")
    print(f"Tunables   : {[t['name'] for t in tunables]}")

    rows = load_csv(args.csv)
    print(f"CSV rows   : {len(rows)}")

    study = build_study(rows, tunables, direction)
    complete_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    skipped_count  = len(rows) - complete_count
    print(f"Complete   : {complete_count}  |  Skipped: {skipped_count}\n")

    if complete_count < 2:
        print("ERROR: Need at least 2 complete trials.")
        sys.exit(1)

    # Compute importance
    print("Computing importance...")
    fanova, mdi = compute_importance(study)

    # Print to terminal
    for label, result in [("fANOVA", fanova), ("MeanDecreaseImpurity", mdi)]:
        if result:
            print(f"\n--- {label} ---")
            for param, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 40)
                print(f"  {param:<55} {score:.4f}  {bar}")

    # Generate plots
    print("\nGenerating plots...")
    plot_files = generate_plots(study, args.output_dir, exp_name)

    # Write importance report HTML
    report_html = build_importance_html(
        experiment_name=exp_name,
        direction=direction,
        total_rows=len(rows),
        complete_count=complete_count,
        skipped_count=skipped_count,
        fanova=fanova,
        mdi=mdi,
        plot_files=plot_files,
        output_dir=args.output_dir,
    )
    report_path = os.path.join(args.output_dir, "importance_report.html")
    with open(report_path, "w") as f:
        f.write(report_html)
    print(f"\n[OK] importance_report.html")
    print(f"\nAll output written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
