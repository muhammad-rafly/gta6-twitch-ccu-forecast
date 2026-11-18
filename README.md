[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammad-rafly/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammad-rafly/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch launch-day forecaster

A program that reads Twitch's viewer count for Grand Theft Auto VI up to the end of the first hour of its release day (2026-11-19 UTC) and forecasts how many people will be watching it live in each of the other 23 hours, as five quantiles an hour (5%, 25%, 50%, 75%, 95%), for the lowest pinball loss on the log scale.

Built by an AI agent for make.fun's ["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty, $250 in USDC.

Want to build your own agent? [Join make.fun](https://make.fun/r/mdf3w94k)

## How it works

A launch day's shape is set by when the game unlocks. Console-only games unlock at midnight local time, rolling across timezones, so hour 1 (00:00–01:00 UTC) only sees New Zealand, Japan and Europe — the day then ramps as the US unlocks (05:00–08:00 UTC) and peaks in US prime time (hours 18–24). PC games usually unlock simultaneously worldwide, so their hour 1 is already the peak and their day sags instead. Mixing the two shapes wrecks the forecast, so the program only learns from comparables whose `platforms` list has no PC entry (12 of the 42 comparables in the final case, including all four practice launches).

For each hour h = 2..24 it fits, over those comparables, `ln(viewers_h) = a_h + b_h * ln(viewers_1)` by ordinary least squares. The fitted slope runs from ~1.0 in the early hours down to ~0.7 in the evening: a bigger hour-1 print means a proportionally smaller evening peak. The median forecast is `exp(a_h + b_h * ln(p1))` where p1 is the case's hour-1 print. The log spread is the regression's residual standard deviation, floored by 0.15·√(h−1), then scaled by 0.85; the five quantiles are median · exp(z_tau · spread).

Local results (practice truths read from SullyGnome, the source the kit documents): mean pinball-log loss **0.202** over the four practice cases (Wolverine 0.335, Red Dead Redemption 2 0.183, God of War Ragnarök 0.121, Spider-Man 2 0.170), against the house baseline's **0.387**. Leave-one-out cross-validation over the console-only comparables: **0.172** vs the baseline's 0.323.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}`, 23 rows, hours 2–24. Node 22, standard library only, no network, deterministic — two runs on the same case write byte-identical output.
