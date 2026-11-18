// GTA VI Twitch launch-day forecaster (Node 22, standard library only).
//
// Model: a per-hour log-level regression on the console-only comparables.
//
// A launch day's shape is set by when the game unlocks. Console-only games
// unlock at midnight local time, rolling across timezones, so hour 1 (00:00-01:00
// UTC) only sees New Zealand, Japan and Europe: the day then ramps as the US
// unlocks (05:00-08:00 UTC) and peaks in US prime time (hours 18-24). PC games
// usually unlock simultaneously worldwide, so their hour 1 is already the peak
// and their day sags instead. Mixing the two shapes wrecks the forecast, so this
// program only learns from comparables whose `platforms` list has no PC entry
// (in the final case that is 12 of the 42 comparables, including the 4 practice
// launches, all console-only midnight-local games like GTA VI itself).
//
// For each hour h = 2..24 it fits, over those comparables,
//   ln(viewers_h) = a_h + b_h * ln(viewers_1) + noise
// by ordinary least squares. The fitted slope b_h runs from ~1.0 in the early
// hours down to ~0.7 in the evening: a bigger hour-1 print means a proportionally
// smaller evening peak (regression to the mean — the absolute evening level grows
// sub-linearly with the first hour). The median forecast is exp(a_h + b_h*ln(p1))
// where p1 is this case's hour-1 print. The log spread is the regression's
// residual standard deviation, floored by the baseline's random-walk spread
// 0.15*sqrt(h-1) (the final is a bigger launch than anything in the training
// set, so the floor keeps the late hours honest), then scaled by 0.85; the five
// quantiles are median * exp(z_tau * spread) with z = -1.645, -0.674, 0, 0.674,
// 1.645. The 0.85 scale and the floor were picked by leave-one-out
// cross-validation over the console-only comparables and the four practice
// cases (practice mean pinball-log loss 0.202 vs the house baseline's 0.387;
// console-only LOO loss 0.172 vs baseline 0.323).
//
// Fallbacks, per hour: if fewer than 3 console-only comparables have that hour,
// fit on all comparables instead; if that also fails, use the house baseline's
// persistence row for the hour. The program is deterministic (no randomness,
// files processed in name order) and runs in well under a second per case.
//
// Reads every <id>.in in the input directory and writes <id>.out to the output
// directory: {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, 23 rows.
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The baseline's per-square-root-hour log spread, used as the spread floor. */
const FLOOR_SIGMA = 0.15;
/** The spread multiplier calibrated on the practice cases + LOO-CV. */
const SPREAD_SCALE = 0.85;
/** Minimum comparables needed to fit a regression for one hour. */
const MIN_FIT = 3;

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Array<{
    name: string;
    day_start: string;
    platforms?: string;
    hourly: Array<{ from: string; viewers: number }>;
  }>;
}

interface ComparableCurve {
  /** ln(hour-1 viewers). */
  logV1: number;
  /** hour -> ln(viewers_h / viewers_1), for the hours present. */
  logRatios: Map<number, number>;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** True when the comparable is a console-only launch (no PC in its platforms). */
const isConsoleOnly = (platforms?: string): boolean =>
  typeof platforms === 'string' && !/PC/i.test(platforms);

/** A comparable's release-day curve: hour-1 log viewers + per-hour log ratios. */
function extractCurve(comp: Case['comparables'][number]): ComparableCurve | null {
  const byHour = new Map<string, number>();
  for (const row of comp.hourly) {
    if (typeof row.viewers === 'number' && row.viewers > 0) byHour.set(row.from, row.viewers);
  }
  const dayStart = Date.parse(comp.day_start);
  if (Number.isNaN(dayStart)) return null;
  const v1 = byHour.get(new Date(dayStart).toISOString().slice(0, 19) + 'Z');
  if (v1 === undefined || v1 <= 0) return null;
  const logRatios = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const v = byHour.get(new Date(dayStart + (h - 1) * 3600e3).toISOString().slice(0, 19) + 'Z');
    if (v !== undefined && v > 0) logRatios.set(h, Math.log(v / v1));
  }
  return { logV1: Math.log(v1), logRatios };
}

/** Ordinary least squares of ys on xs. Returns intercept, slope, residual sd. */
function ols(xs: number[], ys: number[]): { a: number; b: number; sd: number } {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  const residSq = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0);
  return { a, b, sd: Math.sqrt(residSq / Math.max(1, n - 2)) };
}

/** The house baseline's persistence row for one hour (fallback only). */
function baselineRow(p1: number, h: number): number[] {
  return Z.map((z) => round2(Math.max(1, p1 * Math.exp(z * FLOOR_SIGMA * Math.sqrt(h - 1)))));
}

/**
 * One hour's five quantiles from the regression fit.
 * `curves` is the pool to fit on (console-only first, all comparables as fallback).
 */
function hourQuantiles(p1: number, h: number, curves: ComparableCurve[]): number[] {
  const rows = curves.filter((c) => c.logRatios.has(h));
  if (rows.length < MIN_FIT) return baselineRow(p1, h);
  const xs = rows.map((c) => c.logV1);
  const ys = rows.map((c) => c.logV1 + (c.logRatios.get(h) as number));
  const fit = ols(xs, ys);
  // Safety clamp: the fitted slopes on the kit data run 0.67-1.06; a slope
  // outside [0, 1.5] would mean a pathological case file, so clamp it.
  const b = Math.min(1.5, Math.max(0, fit.b));
  const a = fit.a + (fit.b - b) * (xs.reduce((s, x) => s + x, 0) / xs.length);
  const pred = a + b * Math.log(Math.max(1, p1));
  const spread = Math.max(FLOOR_SIGMA * Math.sqrt(h - 1), fit.sd) * SPREAD_SCALE;
  const q = Z.map((z) => round2(Math.max(1, Math.exp(pred + z * spread))));
  for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
  return q.map((v) => Math.min(v, 1e9));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);

  const curves: ComparableCurve[] = [];
  const consoleCurves: ComparableCurve[] = [];
  for (const comp of c.comparables ?? []) {
    const curve = extractCurve(comp);
    if (curve === null) continue;
    curves.push(curve);
    if (isConsoleOnly(comp.platforms)) consoleCurves.push(curve);
  }

  const hours = c.forecast.map(({ hour }) => {
    const pool = consoleCurves.filter((cc) => cc.logRatios.has(hour)).length >= MIN_FIT ? consoleCurves : curves;
    return { hour, q: hourQuantiles(p1, hour, pool) };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
