[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/auzacrypto/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/auzacrypto/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecaster

`main.py` (Python 3.13, standard library only) forecasts the concurrent-viewer
count of the "Grand Theft Auto VI" Twitch category for hours 2-24 of release
day, as five quantiles (5%, 25%, 50%, 75%, 95%) per hour, scored by pinball
loss on the log scale. It reads one case file (hour 1's count, the day's
earlier history, and 38-42 past big launches' hourly curves as comparables)
and writes one forecast file per case. No network, no clock, deterministic.

## The model

A launch day's shape is close to scale-free, so the model works with each
comparable launch's log-ratios `r(h) = ln(viewers(h) / viewers(1))` over its
release day. The forecast is the case's hour-1 print times `exp(Q_tau(h))`,
where `Q_tau(h)` blends, 50/50 in log space, the weighted empirical quantile
of the ratios and a weighted normal fit. Weights are a Gaussian kernel
(bandwidth 0.45) on the log10 distance between the comparable's and the
case's hour-1 counts: launches that opened at a similar scale share the
case's unlock geometry — how much of the world is already playing at
00:00 UTC — which is what shapes the day.

The neighbour pool keeps only clean launches: comparables whose unlock notes
mark an early-access tier or a weak tier are excluded (an audience that could
already play for days draws a different day-one curve; GTA VI and the four
practice launches had no early access), as is the GTA VI category's own
pre-release history, which is not a launch. Fallbacks keep every case valid:
a too-small clean pool re-admits the other launches, and with no usable
comparable at all the forecast is the persistence shape (flat at hour 1,
spread widening like a random walk).

## Validation

The contest has no local grader (the server grades), so the model was tuned
and measured by leave-one-out cross-validation over the 37 usable comparable
launches in the kit's practice cases: each launch in turn is forecast from
the others and scored with the contest's pinball loss. On the targets that
resemble this contest's cases (rolling-midnight unlocks opening at 10k+
viewers in hour 1 — the practice four and GTA VI are of this kind) the model
scores **0.248** against the house persistence baseline's **0.370** on the
same harness (the baseline's stated practice loss is ~0.387, which the
harness reproduces). Over all 37 launches the scores are 0.414 against 0.930.
