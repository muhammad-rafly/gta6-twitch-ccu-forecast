// GTA VI Twitch launch-day forecaster (Node 22, standard library only).
//
// Model v2: a per-hour log-level regression on the console-only comparables,
// blended toward a broader fit and made robust.
//
// A launch day's shape is set by when the game unlocks. Console-only games
// unlock at midnight local time, rolling across timezones, so hour 1 (00:00-01:00
// UTC) only sees New Zealand, Japan and Europe: the day then ramps as the US
// unlocks (05:00-08:00 UTC) and peaks in US prime time (hours 18-24). PC games
// usually unlock simultaneously worldwide, so their hour 1 is already the peak
// and their day sags instead. The console-only pool is therefore the right
// population for GTA VI — but it is only 6 curves in a practice case (10 in the
// final), so its per-hour fits are noisy. v2 fixes that two ways, both validated
// by leave-one-out cross-validation over the console-only comparables and by the
// four practice cases:
//   1. The console fit is a Huber robust regression (iteratively reweighted
//      least squares, k = 1.345 * MAD), so one odd curve cannot drag a fit.
//   2. The median is blended 60/40 toward a second OLS fit on the BROADER pool
//      of comparables with hour-1 viewers >= 10,000 (all platforms). That pool
//      is 17 curves in a practice case (21 in the final): big enough to pin the
//      slope down, while the 10,000 floor keeps out the tiny categories whose
//      astronomical ratios are category-discovery artifacts, not launch ramps.
//      The blend is shrinkage toward a more stable fit, in the spirit of the
//      winning entries of the finished make.fun forecast bounties (empirical
//      quantiles, recency weighting, Laplace fans and shrinkage were all
//      studied; the blend is what actually helped on OUR metrics — see below).
//      The console fit keeps 40% of the voice, so the midnight-rolling shape
//      still drives the forecast, and the evening median still grows smoothly
//      with the hour-1 print (h20 ratio 0.82x at a 875k first hour, against the
//      only mega-launch observation in the pool, ELDEN RING's 0.90x).
//
// For each hour h = 2..24 it fits, over the console comparables,
//   ln(viewers_h) = a_h + b_h * ln(viewers_1) + noise
// by Huber regression. The median forecast blends that fit's prediction with
// the broad-pool fit's prediction (60% broad, 40% console, in log space). The
// log spread is the console fit's residual standard deviation, floored by
// 0.10*sqrt(h-1) (the final is a bigger launch than anything in the training
// set, so the floor keeps the late hours honest), then scaled by 0.85; the five
// quantiles are median * exp(z_tau * spread) with z = -1.645, -0.674, 0, 0.674,
// 1.645. Validated locally (practice truths from SullyGnome, the source the kit
// documents): practice mean pinball-log loss 0.172 vs the v1 model's 0.193 and
// the house baseline's 0.387; console-only LOO loss 0.165 (n=6) and 0.166 (n=10
// on the final-like pool) vs v1's 0.192 / 0.179.
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
/** The house baseline's per-square-root-hour log spread (fallback rows only). */
const BASELINE_SIGMA = 0.15;
/** The spread floor: the final is a bigger launch than the training set, so
    the floor keeps the late hours honest. Calibrated by LOO-CV + practice. */
const SPREAD_FLOOR = 0.10;
/** The spread multiplier calibrated on the practice cases + LOO-CV. */
const SPREAD_SCALE = 0.85;
/** Minimum comparables needed to fit a regression for one hour. */
const MIN_FIT = 3;
/** Weight of the broad-pool fit in the blended median (0.6 broad, 0.4 console). */
const BLEND_WEIGHT = 0.6;
/** The broad pool keeps only comparables with hour-1 viewers at or above this:
    below it the ratios are category-discovery artifacts, not launch ramps. */
const BLEND_MIN_V1 = 10000;

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
  // Decontamination: a launch whose pre-release peak already exceeds its
  // release-day peak was playable before its day_start (early access), so its
  // "release day" curve is not a clean launch-day shape — drop it. Only
  // apply when both sides are well sampled (sparse SullyGnome history could
  // otherwise misfire): at least 5 points in the 14 days before day_start
  // and at least 12 points on the release day; otherwise keep the comparable.
  let preMax = 0;
  let preN = 0;
  let dayMax = 0;
  let dayN = 0;
  for (const [from, viewers] of byHour) {
    const t = Date.parse(from);
    if (Number.isNaN(t)) continue;
    if (t >= dayStart - 14 * 86400e3 && t < dayStart) {
      preN++;
      if (viewers > preMax) preMax = viewers;
    } else if (t >= dayStart && t < dayStart + 24 * 3600e3) {
      dayN++;
      if (viewers > dayMax) dayMax = viewers;
    }
  }
  if (preN >= 5 && dayN >= 12 && preMax > dayMax) return null;
  const v1 = byHour.get(new Date(dayStart).toISOString().slice(0, 19) + 'Z');
  if (v1 === undefined || v1 <= 0) return null;
  const logRatios = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const v = byHour.get(new Date(dayStart + (h - 1) * 3600e3).toISOString().slice(0, 19) + 'Z');
    if (v !== undefined && v > 0) logRatios.set(h, Math.log(v / v1));
  }
  return { logV1: Math.log(v1), logRatios };
}

/** Ordinary least squares of ys on xs (optionally weighted). Returns intercept,
    slope, residuals and their standard deviation. */
function ols(xs: number[], ys: number[], ws: number[] | null = null): { a: number; b: number; resid: number[]; sd: number } {
  const n = xs.length;
  const wsum = ws === null ? n : ws.reduce((s, w) => s + w, 0);
  const mx = (ws === null ? xs.reduce((s, x) => s + x, 0) : xs.reduce((s, x, i) => s + (ws as number[])[i] * x, 0)) / wsum;
  const my = (ws === null ? ys.reduce((s, y) => s + y, 0) : ys.reduce((s, y, i) => s + (ws as number[])[i] * y, 0)) / wsum;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    const w = ws === null ? 1 : ws[i];
    sxy += w * (xs[i] - mx) * (ys[i] - my);
    sxx += w * (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  const resid = ys.map((y, i) => y - (a + b * xs[i]));
  const rss = resid.reduce((s, r, i) => s + (ws === null ? 1 : ws[i]) * r * r, 0);
  return { a, b, resid, sd: Math.sqrt(rss / Math.max(1, n - 2)) };
}

/** Median of an array (helper for the Huber scale). */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Huber robust regression: iteratively reweighted least squares with
    k = 1.345 * MAD, so one odd curve cannot drag the fit. Same return shape
    as ols; the reported sd is the final weighted residual sd. */
function huber(xs: number[], ys: number[]): { a: number; b: number; resid: number[]; sd: number } {
  let fit = ols(xs, ys);
  for (let it = 0; it < 10; it++) {
    const med = median(fit.resid);
    const s = 1.4826 * median(fit.resid.map((r) => Math.abs(r - med))) || 1e-6;
    const k = 1.345 * s;
    const ws = fit.resid.map((r) => (Math.abs(r) <= k ? 1 : k / Math.abs(r)));
    const next = ols(xs, ys, ws);
    if (Math.abs(next.b - fit.b) < 1e-9) {
      fit = next;
      break;
    }
    fit = next;
  }
  return fit;
}

/** The house baseline's persistence row for one hour (fallback only). */
function baselineRow(p1: number, h: number): number[] {
  return Z.map((z) => round2(Math.max(1, p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1)))));
}

/**
 * One hour's five quantiles from the regression fits.
 * `curves` is the pool to fit on (console-only first, all comparables as fallback);
 * `broad` is the broad pool (all comparables with hour-1 viewers >= BLEND_MIN_V1)
 * whose OLS fit the median is blended toward.
 */
function hourQuantiles(p1: number, h: number, curves: ComparableCurve[], broad: ComparableCurve[]): number[] {
  const rows = curves.filter((c) => c.logRatios.has(h));
  if (rows.length < MIN_FIT) return baselineRow(p1, h);
  const xs = rows.map((c) => c.logV1);
  const ys = rows.map((c) => c.logV1 + (c.logRatios.get(h) as number));
  const fit = huber(xs, ys);
  // Safety clamp: the fitted slopes on the kit data run 0.67-1.06; a slope
  // outside [0, 1.5] would mean a pathological case file, so clamp it.
  const b = Math.min(1.5, Math.max(0, fit.b));
  const a = fit.a + (fit.b - b) * (xs.reduce((s, x) => s + x, 0) / xs.length);
  let pred = a + b * Math.log(Math.max(1, p1));
  // Blend toward the broad-pool OLS fit (shrinkage toward a more stable fit).
  const broadRows = broad.filter((c) => c.logRatios.has(h));
  if (broadRows.length >= MIN_FIT) {
    const xsB = broadRows.map((c) => c.logV1);
    const ysB = broadRows.map((c) => c.logV1 + (c.logRatios.get(h) as number));
    const fitB = ols(xsB, ysB);
    const bB = Math.min(1.5, Math.max(0, fitB.b));
    const aB = fitB.a + (fitB.b - bB) * (xsB.reduce((s, x) => s + x, 0) / xsB.length);
    const predB = aB + bB * Math.log(Math.max(1, p1));
    pred = (1 - BLEND_WEIGHT) * pred + BLEND_WEIGHT * predB;
  }
  const spread = Math.max(SPREAD_FLOOR * Math.sqrt(h - 1), fit.sd) * SPREAD_SCALE;
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
  const broadCurves: ComparableCurve[] = [];
  for (const comp of c.comparables ?? []) {
    const curve = extractCurve(comp);
    if (curve === null) continue;
    curves.push(curve);
    if (isConsoleOnly(comp.platforms)) consoleCurves.push(curve);
    if (Math.exp(curve.logV1) >= BLEND_MIN_V1) broadCurves.push(curve);
  }

  const hours = c.forecast.map(({ hour }) => {
    const pool = consoleCurves.filter((cc) => cc.logRatios.has(hour)).length >= MIN_FIT ? consoleCurves : curves;
    return { hour, q: hourQuantiles(p1, hour, pool, broadCurves) };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
