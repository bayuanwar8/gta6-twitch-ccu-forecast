"""Evaluate the forecaster in main.py against the house baseline.

Two validations:

1. LEAVE-ONE-OUT over the kit's comparables (the real validation). Every
   comparable carries its own full release-day hourly curve, so for each one
   we forecast it from the OTHER comparables (through the real forecast_case
   code path) and score the pinball loss on the log scale against its actual
   hours 2..24, next to the house baseline's loss on the same hours.

2. PRACTICE CASES (p00..p03): run the real forecast and check the output is
   valid and the median shape is sane. No loss is computable here: the kit
   holds no actuals for the practice launches' hours 2..24 (their history
   stops at the checkpoint, and every practice case's comparables exclude all
   four practice launches -- verified, not assumed).

Pinball loss (log scale), per (hour, level): y = log(actual), q = log(forecast)
  loss = (y - q) * tau   if y >= q
       = (q - y) * (1 - tau)  otherwise
averaged over all hours x levels that have an actual.

Usage: python3 tools/eval.py [kit-dir]
  kit-dir defaults to ../kit (sibling of the repo) -- pass the kit path.
"""

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from main import (  # noqa: E402
    LEVELS,
    collect_log_ratios,
    forecast_case,
    release_day_viewers,
)

Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15


def baseline_quantiles(p1, h):
    """House baseline (starter): persistence with a widening random-walk spread."""
    return [p1 * math.exp(z * SIGMA * math.sqrt(h - 1)) for z in Z]


def pinball_loss(actual, quantiles, levels=LEVELS):
    """Mean pinball loss on the log scale over one hour's five quantiles."""
    y = math.log(actual)
    total = 0.0
    for tau, q in zip(levels, quantiles):
        lq = math.log(q)
        total += (y - lq) * tau if y >= lq else (lq - y) * (1 - tau)
    return total / len(levels)


def score_forecast(forecast, actual_by_hour):
    """Mean pinball loss over all scored (hour, level) pairs."""
    losses = []
    by_hour = {row["hour"]: row["q"] for row in forecast["hours"]}
    for h in sorted(actual_by_hour):
        if h in by_hour and actual_by_hour[h] > 0:
            losses.append(pinball_loss(actual_by_hour[h], by_hour[h]))
    if not losses:
        return None
    return sum(losses) / len(losses)


def score_baseline(p1, actual_by_hour):
    losses = []
    for h in sorted(actual_by_hour):
        if h >= 2 and actual_by_hour[h] > 0:
            losses.append(pinball_loss(actual_by_hour[h], baseline_quantiles(p1, h)))
    if not losses:
        return None
    return sum(losses) / len(losses)


def load_cases(kit_dir):
    cases = {}
    for path in sorted((kit_dir / "input").glob("p*.in")):
        cases[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return cases


def main():
    kit_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "twitch-ccu-01" / "gta6-twitch-ccu-forecast-01-kit"
    cases = load_cases(kit_dir)
    print(f"practice cases: {sorted(cases)}")

    # Are the comparable sets identical across practice cases?
    sets = {k: {c["name"] for c in v["comparables"]} for k, v in cases.items()}
    first = next(iter(sets.values()))
    print(f"comparable sets identical across cases: {all(s == first for s in sets.values())} (n={len(first)})")

    # ---- 1. Leave-one-out over the comparables of p00 -------------------
    comps = cases["p00"]["comparables"]
    print(f"\n=== LEAVE-ONE-OUT over {len(comps)} comparables ===")
    rows = []
    for comp in sorted(comps, key=lambda c: c["name"]):
        day = release_day_viewers(comp)
        p1 = day.get(1)
        if not p1:
            print(f"{comp['name']:45s} SKIP (no hour-1 print)")
            continue
        actual = {h: v for h, v in day.items() if h >= 2}
        if len(actual) < 12:
            print(f"{comp['name']:45s} SKIP (only {len(actual)} actual hours)")
            continue
        others = [c for c in comps if c["name"] != comp["name"]]
        pseudo = {
            "levels": LEVELS,
            "known": [{"hour": 1, "viewers": p1}],
            "forecast": [{"hour": h} for h in range(2, 25)],
            "comparables": others,
        }
        forecast = forecast_case(pseudo, comp["name"])
        model_loss = score_forecast(forecast, actual)
        base_loss = score_baseline(p1, actual)
        rows.append((comp["name"], model_loss, base_loss, len(actual)))
        flag = "WIN " if model_loss < base_loss else "LOSE"
        print(f"{comp['name']:45s} model={model_loss:.4f} baseline={base_loss:.4f} {flag} (n={len(actual)}h)")

    if rows:
        m = sum(r[1] for r in rows) / len(rows)
        b = sum(r[2] for r in rows) / len(rows)
        wins = sum(1 for r in rows if r[1] < r[2])
        print(f"\nLOO MEAN: model={m:.4f} baseline={b:.4f} "
              f"improvement={(b - m) / b * 100:.1f}% | model wins {wins}/{len(rows)} launches")

    # ---- 2. Practice cases: run + sanity-check the shape ----------------
    print("\n=== PRACTICE CASES (no actuals in kit: shape + validity only) ===")
    for pid in sorted(cases):
        case = cases[pid]
        forecast = forecast_case(case, pid)
        p1 = max(1.0, float(next(k for k in case["known"] if k["hour"] == 1)["viewers"]))
        ratios = [f"{row['q'][2] / p1:.2f}" for row in forecast["hours"]]
        valid = (
            len(forecast["hours"]) == 23
            and [r["hour"] for r in forecast["hours"]] == list(range(2, 25))
            and all(len(r["q"]) == 5 and all(q > 0 and q <= 1e9 for q in r["q"])
                    and r["q"] == sorted(r["q"]) for r in forecast["hours"])
        )
        print(f"{pid} {case['game']['name']:28s} p1={p1:,.0f} valid={valid}")
        print(f"   q50 ratio vs hour-1, h=2..24: {' '.join(ratios)}")


if __name__ == "__main__":
    main()
