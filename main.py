"""gta6-twitch-ccu-forecast-01: forecast a launch day's Twitch CCU curve, hours 2-24.

Model (fit offline by leave-one-out cross-validation over the comparable launches
shipped in every case; see README.md):

  A launch day's shape is, to first order, scale-free: work with each comparable
  launch's log-ratios r(h) = ln(viewers(h) / viewers(1)) over its release day
  (UTC), h = 2..24. The forecast for a case is its own hour-1 print p1 times the
  exponential of a per-hour quantile of those ratios:

    q_tau(h) = p1 * exp( Q_tau(h) ),

  where Q_tau(h) blends two estimates of the tau-quantile of r(h), 50/50 in log
  space: the weighted empirical quantile and a weighted normal fit (weighted
  mean + z_tau * weighted sd). Weights are a Gaussian kernel on the distance
  between the comparable's and the case's hour-1 counts, in log10 with bandwidth
  0.45: launches that opened at a similar scale share the case's unlock geometry
  (how much of the world is already playing at 00:00 UTC), which is what shapes
  the day.

  The neighbour pool is restricted to clean launches: the comparables whose
  unlock notes carry an early-access tier ("Tier: early-access-contaminated") or
  a weak tier ("Tier: weak") are left out, because an audience that could already
  play for days draws a different day-one curve, and this contest's launches
  (the practice four, and Grand Theft Auto VI) had no early access. The Grand
  Theft Auto VI category's own pre-release history (marked "Not a launch") is
  not a launch and is left out too. If fewer than 8 clean launches remain, the
  pool falls back to every launch, and if no comparable is usable at all, the
  forecast falls back to the house persistence shape (flat at p1, spread
  0.15 * sqrt(h - 1)), so a case always gets a valid forecast.

  Quantiles are sorted non-decreasing per hour and clamped to (0, 1e9].

Runs as the grader runs it: `python3 -P -s /bot/main.py`, standard library only,
no network; reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output). Deterministic: fixed
iteration order, no randomness, no clock.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (as the starter does).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# Gaussian kernel bandwidth on the log10 hour-1 distance between launches.
BANDWIDTH = 0.45
# Weight of the empirical quantile in the log-space blend (the rest is the normal fit).
EMPIRICAL_SHARE = 0.5
# Fewer clean launches than this and the pool falls back to every launch.
MIN_CLEAN_POOL = 8


def parse_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as the starter rounds)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def day_curve(comp: dict) -> tuple[float, dict[int, float]] | None:
    """(hour-1 viewers, {hour: ln(viewers(hour)/viewers(1))}) for a comparable's
    release day, or None when the day has no usable first hour."""
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return None
    viewers = {}
    for point in comp.get("hourly") or []:
        try:
            t = parse_ts(point["from"])
            v = point["viewers"]
        except (KeyError, ValueError):
            continue
        if v is None or v <= 0:
            continue
        offset = t - start
        if offset % 3600 == 0:
            h = int(offset // 3600) + 1
            if 1 <= h <= 24:
                viewers[h] = v
    if 1 not in viewers:
        return None
    base = math.log(viewers[1])
    ratios = {h: math.log(v) - base for h, v in viewers.items() if h >= 2}
    return float(viewers[1]), ratios


def pool_tier(comp: dict) -> str:
    unlock = comp.get("unlock") or ""
    if unlock.startswith("Not a launch"):
        return "skip"
    if "Tier: early-access-contaminated" in unlock:
        return "early"
    if "Tier: weak" in unlock:
        return "weak"
    return "clean"


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Linear-interpolation weighted quantile (midpoint rule on the weighted CDF)."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    cum = []
    s = 0.0
    for _, w in pairs:
        s += w / total
        cum.append(s)
    pts = [(cum[i] - (pairs[i][1] / total) / 2.0, pairs[i][0]) for i in range(len(pairs))]
    if tau <= pts[0][0]:
        return pairs[0][0]
    if tau >= pts[-1][0]:
        return pairs[-1][0]
    for i in range(1, len(pts)):
        if tau <= pts[i][0]:
            p0, x0 = pts[i - 1]
            p1_, x1 = pts[i]
            if p1_ == p0:
                return x1
            return x0 + (x1 - x0) * (tau - p0) / (p1_ - p0)
    return pairs[-1][0]


def build_pool(case: dict) -> list[tuple[float, dict[int, float]]]:
    """[(hour-1 viewers, {hour: log-ratio})] for the case's usable comparables."""
    game_name = (case.get("game") or {}).get("name")
    day_start = case.get("day_start")
    clean, rest = [], []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name") or ""):
        # Never learn from the case's own launch, if it were ever included.
        if comp.get("name") == game_name and comp.get("day_start") == day_start:
            continue
        tier = pool_tier(comp)
        if tier == "skip":
            continue
        curve = day_curve(comp)
        if not curve:
            continue
        p1, ratios = curve
        (clean if tier == "clean" else rest).append((p1, ratios))
    if len(clean) >= MIN_CLEAN_POOL:
        return clean
    return clean + rest


def hour_log_quantiles(pool: list[tuple[float, dict[int, float]]], log10_p1: float,
                       h: int) -> list[float] | None:
    pairs = []
    for p1, ratios in pool:
        if h in ratios:
            d = abs(math.log10(p1) - log10_p1)
            w = math.exp(-0.5 * (d / BANDWIDTH) ** 2)
            pairs.append((ratios[h], w))
    if not pairs:
        return None
    empirical = [weighted_quantile(pairs, tau) for tau in LEVELS]
    total = sum(w for _, w in pairs)
    mu = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / total
    sd = math.sqrt(var)
    normal = [mu + z * sd for z in Z]
    blended = [EMPIRICAL_SHARE * e + (1.0 - EMPIRICAL_SHARE) * n
               for e, n in zip(empirical, normal)]
    return sorted(blended)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    pool = build_pool(case)
    log10_p1 = math.log10(p1)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        log_qs = hour_log_quantiles(pool, log10_p1, h) if pool else None
        if log_qs is None:
            # House persistence shape: flat at p1 with a random-walk spread.
            log_qs = [z * 0.15 * math.sqrt(h - 1) for z in Z]
        qs = []
        prev = 0.0
        for lq in log_qs:
            q = p1 * math.exp(lq)
            q = min(max(q, 1e-6), 1e9)
            q = max(q, prev)  # non-decreasing, as the format requires
            prev = q
            qs.append(round2(q))
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
