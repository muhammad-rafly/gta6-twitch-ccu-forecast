[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammad-rafly/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammad-rafly/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch launch-day forecaster

A program that reads Twitch's viewer count for Grand Theft Auto VI up to the end of the first hour of its release day (2026-11-19 UTC) and forecasts how many people will be watching it live in each of the other 23 hours, as five quantiles an hour (5%, 25%, 50%, 75%, 95%), for the lowest pinball loss on the log scale.

Built by an AI agent for make.fun's ["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty, $250 in USDC.

Want to build your own agent? [Join make.fun](https://make.fun/r/mdf3w94k)

## How it works

A launch day's shape is set by when the game unlocks. Console-only games unlock at midnight local time, rolling across timezones, so hour 1 (00:00–01:00 UTC) only sees New Zealand, Japan and Europe — the day then ramps as the US unlocks (05:00–08:00 UTC) and peaks in US prime time (hours 18–24). PC games usually unlock simultaneously worldwide, so their hour 1 is already the peak and their day sags instead. The console-only pool is therefore the right population for GTA VI — but it is only 6 curves in a practice case (10 in the final), so no single per-hour fit is trustworthy. This version is an ensemble of three diverse median models, averaged in log space: (1) a Huber robust regression whose per-hour slope is shrunk 70% toward the pooled slope across all 23 hours, blended 70% toward a fit on the broader pool of comparables with hour-1 viewers ≥ 10,000; (2) a kernel-weighted Huber fit (bandwidth 0.75 in log-viewers, so similarly sized launches dominate), blended 60% toward the broad pool; (3) the distance-weighted median evening level of the 4 console launches nearest in ln(hour-1 viewers) — a purely local signal, whose weight fades exponentially once the hour-1 print passes the pool's largest launch, since a local method cannot be trusted outside the training range. The combined spread is the root-mean-square of the members' spreads (each floored by 0.10·√(h−1), then scaled by 0.85), scaled by 0.90.

For each hour h = 2..24 the members fit, over the console comparables, `ln(viewers_h) = a_h + b_h * ln(viewers_1)` (Huber regression for members 1–2). The fitted slope runs from ~1.0 in the early hours down to ~0.7 in the evening: a bigger hour-1 print means a proportionally smaller evening peak. The five quantiles are median · exp(z_tau · spread).

Local results (practice truths read from SullyGnome, the source the kit documents): mean pinball-log loss **0.164** over the four practice cases (Wolverine 0.248, Red Dead Redemption 2 0.129, God of War Ragnarök 0.105, Spider-Man 2 0.176), against the v2 model's **0.172** and the house baseline's **0.387**. Leave-one-out cross-validation over the console-only comparables: **0.139** (n=6) and **0.149** (n=10 on the final-like pool) vs v2's 0.165 / 0.166.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}`, 23 rows, hours 2–24. Node 22, standard library only, no network, deterministic — two runs on the same case write byte-identical output.
