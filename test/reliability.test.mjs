import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as RL from '../core/reliability.mjs';
import { analyze, heldOut, backtest } from '../core/analysis.mjs';

const fleet = JSON.parse(readFileSync(new URL('../data/fleet.json', import.meta.url)));

test('Poisson GLM recovers known coefficients', () => {
  const rng = RL.mulberry32(4), N = RL.normalSampler(rng);
  const Z = [], y = [], e = [];
  for (let i = 0; i < 20000; i++) {
    const z = [N(), N()], mu = Math.exp(-2 + 0.7 * z[0] - 0.4 * z[1]);
    let k = 0, p = 1, L = Math.exp(-mu); do { k++; p *= rng(); } while (p > L);
    Z.push(z); y.push(k - 1); e.push(1);
  }
  const b = RL.fitPoissonGLM(Z, y, e, { l2: 0.01 });
  assert.ok(Math.abs(b[0] + 2) < 0.05 && Math.abs(b[1] - 0.7) < 0.05 && Math.abs(b[2] + 0.4) < 0.05, b.join(','));
});

test('goodput model: Young approximation matches the numeric optimum when C is small; goodput is unimodal', () => {
  const C = 4 / 60, R = 25 / 60, L = 1 / 30;
  const t = RL.optimalTau(C, R, L), y = RL.youngTau(C, L);
  assert.ok(Math.abs(t - y) / y < 0.06, `${t} vs ${y}`);
  assert.ok(RL.goodput(t, C, R, L) >= RL.goodput(t * 0.5, C, R, L) && RL.goodput(t, C, R, L) >= RL.goodput(t * 2, C, R, L));
  assert.equal(RL.goodput(1, C, R, 0), 1 / (1 + C));
});

test('simulateRun agrees with the analytic goodput on Poisson failures', () => {
  const rng = RL.mulberry32(8), L = 1 / 20, C = 0.1, R = 0.4, tau = 1.5, H = 200000;
  const fails = []; let t = 0; while (t < H) { t += -Math.log(1 - rng()) / L; fails.push(t); }
  const sim = RL.simulateRun(fails, 0, H, tau, C, R), ana = RL.goodput(tau, C, R, L);
  assert.ok(Math.abs(sim - ana) < 0.01, `${sim} vs ${ana}`);
});

test('gamma sampler has the right mean and variance', () => {
  const rng = RL.mulberry32(2), xs = Array.from({ length: 40000 }, () => RL.gammaSample(rng, 3.5));
  const m = xs.reduce((a, b) => a + b, 0) / xs.length, v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length;
  assert.ok(Math.abs(m - 3.5) < 0.05 && Math.abs(v - 3.5) < 0.15, `${m} ${v}`);
});

test('fleet data matches the public interruption rate it was calibrated to', () => {
  const n = fleet.nodes.reduce((s, x) => s + x.failures.length, 0) + fleet.racks.reduce((s, r) => s + r.rack_failures.length, 0);
  const perGpuDay = n / (fleet.nodes.length * 4 * fleet.days);
  assert.ok(perGpuDay > 0.7 * fleet.per_gpu_day_rate_public && perGpuDay < 1.4 * fleet.per_gpu_day_rate_public, perGpuDay);
});

test('model uses only past telemetry, ranks truly frail trays higher, and is roughly calibrated out of sample', () => {
  const A = analyze(fleet, 120);
  // rank correlation between predicted rate and hidden frailty
  const pairs = fleet.nodes.map((n, i) => [A.nodeRate[i], n.truth_z]);
  const rank = xs => { const o = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]); const r = Array(xs.length); o.forEach(([, i], k) => { r[i] = k; }); return r; };
  const ra = rank(pairs.map(p => p[0])), rb = rank(pairs.map(p => p[1]));
  const n = ra.length, d2 = ra.reduce((s, r, i) => s + (r - rb[i]) ** 2, 0), rho = 1 - 6 * d2 / (n * (n * n - 1));
  assert.ok(rho > 0.5, `spearman ${rho}`);
  const h = heldOut(fleet, A, 60);
  assert.ok(h.skill > 0.05, `deviance skill ${h.skill}`);
  assert.ok(h.observed > 0.7 * h.expected && h.observed < 1.5 * h.expected, `${h.expected} vs ${h.observed}`);
  const top = h.table[4], bottom = h.table[0];
  assert.ok(top.observed > 3 * bottom.observed, 'risk bands separate');
});

test('backtest: planner beats default racks + 1h checkpoints on realized failures', () => {
  const A = analyze(fleet, 120);
  const bt = backtest(fleet, A, { racksNeeded: 32, C: 4 / 60, R: 25 / 60, runDays: 30, starts: 30, defaultTau: 1 });
  assert.ok(bt.mean.planner > bt.mean.baseline + 0.005, JSON.stringify(bt.mean));
  assert.ok(bt.interruptions.planner < bt.interruptions.baseline);
});
