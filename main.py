"""Forecast Twitch concurrent viewers for the GTA VI launch day (UTC hours 2-24).

Model: a weighted log-ratio diurnal shape learned from past big launches.

The case hands us hour 1's viewer count (p1) plus the full hourly curves of
~38 comparable past launches ("comparables"). For each comparable we compute
the log-ratio of every later hour to its own hour 1:

    r(h) = log(viewers(h) / viewers(1))

A launch day is not flat: viewers dip overnight, then climb as regions wake
up and the US comes online in the evening. The ratios capture that shape in a
scale-free way, so it transfers to GTA VI even though GTA VI is far bigger
than any comparable.

Per hour we take the WEIGHTED quantiles of r(h) across comparables, weighting
by similarity to GTA VI (a console-only, single-player, midnight-rolling
launch): weight 3 for console-only + midnight-rolling unlock, 2 for either
one of those, 1 otherwise. The forecast is then

    q_tau(h) = p1 * exp(weighted_quantile_tau(r(h)))

with the five scored levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Fallback: an hour no comparable has data for -- or a case with no
comparables at all -- is forecast with the house baseline's persistence
formula (p1 with a random-walk spread widening in log space), so the entry
degrades to the baseline instead of borrowing another hour's shape or
going flat.

Runs offline on Python 3.13, standard library only, fully deterministic:
every iteration is over sorted lists, no randomness, no dict-order dependence.

I/O contract (same as the contest spec):
  - reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input)
  - writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output):
    {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24,
    each q finite, > 0, <= 1e9, non-decreasing, rounded to 2 decimals half-up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# House baseline (starter) spread, used ONLY as a fallback: persistence at
# hour 1 with a random-walk spread in log space, sigma 0.15 per sqrt hour.
BASELINE_SIGMA = 0.15
BASELINE_Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A comparable only counts if its hour-1 print is a real launch print, not a
# pre-unlock trickle: games whose global unlock falls hours after midnight UTC
# have a near-zero hour 1 and absurd log-ratios. Require hour 1 to be at least
# this fraction of the comparable's own release-day peak.
MIN_HOUR1_FRACTION_OF_PEAK = 0.02

# Output bounds from the contest spec.
MAX_VIEWERS = 1e9


def round_half_up_2(x):
    """Round to 2 decimals, half up; return an int when the result is whole."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def parse_ts(text):
    """Parse a 'YYYY-MM-DDTHH:MM:SSZ' timestamp into an aware datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def comparable_weight(comp):
    """How similar is this past launch to GTA VI? 3 / 2 / 1.

    GTA VI is a console-only, single-player, midnight-rolling launch.
    Console-only means the platform list names no PC / cloud platform.
    Midnight-rolling means the unlock note says midnight local / rolling.
    """
    platforms = (comp.get("platforms") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    console_only = "pc" not in platforms and "stadia" not in platforms
    midnight_rolling = "midnight local" in unlock or "rolling" in unlock
    if console_only and midnight_rolling:
        return 3
    if console_only or midnight_rolling:
        return 2
    return 1


def release_day_viewers(comp):
    """Map UTC hour (1..24) of the comparable's release day to viewers.

    Hour 1 is the hour starting at day_start (00:00 UTC). Rows with no
    viewers, or viewers <= 0, are dropped; missing hours simply stay absent.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            at = parse_ts(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = float(viewers)
    return out


def collect_log_ratios(comparables, exclude_name=None):
    """Per hour, the list of (log-ratio, weight) pairs from the comparables.

    A comparable is skipped when it has no hour-1 print, or when hour 1 is a
    pre-unlock trickle (below MIN_HOUR1_FRACTION_OF_PEAK of its own peak).
    """
    by_hour = {h: [] for h in range(2, 25)}
    for comp in comparables:
        name = comp.get("name")
        if exclude_name is not None and name == exclude_name:
            continue
        day = release_day_viewers(comp)
        p1 = day.get(1)
        if not p1:
            continue
        peak = max(day.values())
        if p1 < MIN_HOUR1_FRACTION_OF_PEAK * peak:
            continue
        weight = comparable_weight(comp)
        for h in range(2, 25):
            v = day.get(h)
            if v:
                by_hour[h].append((math.log(v / p1), weight))
    return by_hour


def weighted_quantile(pairs, tau):
    """Weighted quantile by linear interpolation on cumulative weight.

    Pairs are (value, weight); values are sorted ascending, then we walk the
    cumulative weight until it reaches tau * total weight, interpolating
    linearly inside that point's weight interval.
    """
    pts = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in pts)
    if total <= 0:
        raise ValueError("weighted_quantile needs positive total weight")
    target = tau * total
    cum = 0.0
    prev_value = pts[0][0]
    for value, weight in pts:
        prev_cum = cum
        cum += weight
        if cum >= target:
            if cum == prev_cum:
                return value
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_value + (value - prev_value) * frac
        prev_value = value
    return pts[-1][0]


def clean_quantiles(qs):
    """Enforce the spec on five raw quantiles: finite, > 0, <= 1e9,
    non-decreasing, rounded to 2 decimals half-up."""
    cleaned = []
    floor = 0.01
    for q in qs:
        if not math.isfinite(q) or q <= 0:
            q = floor
        q = min(q, MAX_VIEWERS)
        if cleaned:
            q = max(q, cleaned[-1])
        cleaned.append(round_half_up_2(q))
    # Rounding is monotonic, but re-enforce non-decreasing after rounding.
    for i in range(1, len(cleaned)):
        if cleaned[i] < cleaned[i - 1]:
            cleaned[i] = cleaned[i - 1]
    return cleaned


def hour_forecast(p1, ratios):
    """The five quantiles for one hour, from its (log-ratio, weight) pairs."""
    qs = [p1 * math.exp(weighted_quantile(ratios, tau)) for tau in LEVELS]
    return clean_quantiles(qs)


def persistence_forecast(p1, h):
    """House-baseline fallback for one hour: persistence at p1 with a
    random-walk spread widening in log space (sigma 0.15 per sqrt hour).

    Used only when no comparable has log-ratio data for that hour, or the
    case has no comparables at all. In that situation the honest forecast
    is the baseline itself: it keeps the entry valid and degrades to the
    house score instead of borrowing another hour's shape or going flat.
    """
    qs = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in BASELINE_Z]
    return clean_quantiles(qs)


def forecast_case(case, name="case"):
    """Forecast hours 2..24 of one case: {"hours": [{"hour": h, "q": [...]}]}."""
    levels = case.get("levels") or LEVELS
    if list(levels) != list(LEVELS):
        raise ValueError(f"{name} asks for levels {levels}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    by_hour = collect_log_ratios(case.get("comparables") or [])
    hours = []
    for row in case.get("forecast") or []:
        h = int(row["hour"])
        pairs = by_hour.get(h) or []
        if pairs:
            q = hour_forecast(p1, pairs)
        else:
            # No comparable has log-ratio data for this hour -- or the case
            # has no comparables at all -- so this hour falls back to the
            # house baseline persistence forecast, not a borrowed shape.
            q = persistence_forecast(p1, h)
        hours.append({"hour": h, "q": q})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
