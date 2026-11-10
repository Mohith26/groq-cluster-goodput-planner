// Goodput planner for reserved GPU clusters: calibrated interruption risk per node and rack,
// rack selection, and checkpoint-interval optimization with uncertainty.
// Pure functions, no dependencies. Runs in the browser (ES module) and under `node --test`.
//
//   1. features(node, day)        rolling telemetry summary (14-day window) per node
//   2. fitPoissonGLM(X, y)        L2-regularized Poisson regression by Newton/IRLS: daily interruption rate
//   3. calibrationTable(...)      predicted vs observed interruptions on held-out days (reliability check)
//   4. rackPosterior(...)         Gamma posterior per rack: model rate as prior, rack's own history as data
//   5. goodput(tau, C, R, L)      first-order checkpoint/restart model; optimalTau by golden-section search
//   6. planJob(...)               choose racks, tau, and propagate rate uncertainty to goodput
//   7. simulateRun(...)           replay a job against realized failure times (backtest)

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function normalSampler(rng) {
  return () => { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); };
}
// Marsaglia-Tsang gamma sampler (shape k > 0, rate 1).
export function gammaSample(rng, k) {
  const N = normalSampler(rng);
  if (k < 1) return gammaSample(rng, k + 1) * Math.pow(rng(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = N(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v; const u = rng();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
export function quantile(xs, p) {
  const s = [...xs].sort((a, b) => a - b);
  const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
}

export const FEATURES = ['sbe_ecc', 'nvlink_crc', 'xid_warn', 'pcie_aer', 'temp_excess', 'early_life'];
export const FEATURE_LABEL = {
  sbe_ecc: 'Correctable ECC (SBE) / day', nvlink_crc: 'NVLink CRC errors / day', xid_warn: 'Non-fatal XID events / day',
  pcie_aer: 'PCIe AER errors / day', temp_excess: 'GPU temp above 75°C (°C)', early_life: 'Early life (burn-in decay)',
};

// Rolling 14-day means of the raw telemetry, log1p-compressed for count features.
export function features(tel, day, win = 14) {
  const a = Math.max(0, day - win), n = day - a || 1;
  const m = k => { let s = 0; for (let d = a; d < day; d++) s += tel[k][d]; return s / n; };
  return [Math.log1p(m('sbe_ecc')), Math.log1p(m('nvlink_crc')), Math.log1p(m('xid_warn')), Math.log1p(m('pcie_aer')), m('temp_excess'), Math.exp(-(tel.age0 + day) / 15)];
}

export function standardizer(X) {
  const p = X[0].length, mu = Array(p).fill(0), sd = Array(p).fill(0);
  for (const x of X) for (let j = 0; j < p; j++) mu[j] += x[j] / X.length;
  for (const x of X) for (let j = 0; j < p; j++) sd[j] += (x[j] - mu[j]) ** 2 / X.length;
  for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return { mu, sd, apply: x => x.map((v, j) => (v - mu[j]) / sd[j]) };
}

function solve(A, b) { // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}

// Poisson GLM, log link, exposure in days: y ~ Poisson(exposure * exp(b0 + Z b)). Newton with L2 on b (not b0).
export function fitPoissonGLM(Z, y, exposure, { l2 = 1, iters = 30 } = {}) {
  const p = Z[0].length + 1, beta = Array(p).fill(0);
  const ybar = y.reduce((a, b) => a + b, 0) / exposure.reduce((a, b) => a + b, 0);
  beta[0] = Math.log(Math.max(ybar, 1e-9));
  for (let it = 0; it < iters; it++) {
    const g = Array(p).fill(0), Hm = Array.from({ length: p }, () => Array(p).fill(0));
    for (let i = 0; i < Z.length; i++) {
      const x = [1, ...Z[i]];
      let eta = 0; for (let j = 0; j < p; j++) eta += beta[j] * x[j];
      const mu = exposure[i] * Math.exp(eta);
      for (let j = 0; j < p; j++) { g[j] += (y[i] - mu) * x[j]; for (let k = 0; k < p; k++) Hm[j][k] += mu * x[j] * x[k]; }
    }
    for (let j = 1; j < p; j++) { g[j] -= l2 * beta[j]; Hm[j][j] += l2; }
    const step = solve(Hm, g);
    let mx = 0; for (let j = 0; j < p; j++) { beta[j] += step[j]; mx = Math.max(mx, Math.abs(step[j])); }
    if (mx < 1e-8) break;
  }
  return beta;
}
export const rateOf = (beta, z) => Math.exp(beta[0] + z.reduce((s, v, j) => s + v * beta[j + 1], 0));

// Reliability check: bucket held-out node-windows by predicted interruptions, compare to observed.
export function calibrationTable(pred, obs, bins = 5) {
  const idx = pred.map((p, i) => i).sort((a, b) => pred[a] - pred[b]);
  const out = [];
  for (let b = 0; b < bins; b++) {
    const part = idx.slice(Math.floor(b * idx.length / bins), Math.floor((b + 1) * idx.length / bins));
    const e = part.reduce((s, i) => s + pred[i], 0), o = part.reduce((s, i) => s + obs[i], 0);
    // exact Poisson 90% interval for the observed count (Garwood) via Wilson-Hilferty approximation
    const lo = o === 0 ? 0 : o * (1 - 1 / (9 * o) - 1.645 / (3 * Math.sqrt(o))) ** 3;
    const hi = (o + 1) * (1 - 1 / (9 * (o + 1)) + 1.645 / (3 * Math.sqrt(o + 1))) ** 3;
    out.push({ expected: e, observed: o, lo, hi, n: part.length });
  }
  return out;
}

// Poisson deviance-based skill vs a constant-rate baseline (higher is better; 0 = no skill).
export function devianceSkill(pred, base, obs) {
  const dev = m => obs.reduce((s, y, i) => s + 2 * ((y > 0 ? y * Math.log(y / m[i]) : 0) - (y - m[i])), 0);
  return 1 - dev(pred) / dev(base);
}

// Rack posterior: Gamma(a0 + failures, b0 + exposure) with prior mean = model rate and prior weight k days.
export function rackPosterior(modelRate, failures, exposureDays, k = 60) {
  return { shape: modelRate * k + failures, rate: k + exposureDays, mean: (modelRate * k + failures) / (k + exposureDays) };
}

// First-order checkpoint/restart model (Young/Daly). tau, C, R in hours; L = interruptions per hour.
// Each cycle computes tau then writes a checkpoint of cost C; a failure loses on average half a cycle plus restart R.
export function goodput(tau, C, R, L) {
  const g = (tau / (tau + C)) * (1 - L * (R + (tau + C) / 2));
  return Math.max(0, g);
}
export function optimalTau(C, R, L) {
  let a = 1 / 60, b = 48, phi = (Math.sqrt(5) - 1) / 2;
  let c = b - phi * (b - a), d = a + phi * (b - a);
  for (let i = 0; i < 80; i++) { if (goodput(c, C, R, L) > goodput(d, C, R, L)) b = d; else a = c; c = b - phi * (b - a); d = a + phi * (b - a); }
  return (a + b) / 2;
}
export const youngTau = (C, L) => Math.sqrt(2 * C / L);

// Plan a job on `racksNeeded` racks: rank racks by posterior mean rate, pick the lowest, choose tau,
// and propagate posterior uncertainty in the job's interruption rate to goodput and lost GPU-hours.
export function planJob(racks, racksNeeded, { C, R, gpusPerRack = 72, sims = 2000, seed = 1, choose = 'planner', baseline = null } = {}) {
  const sorted = [...racks].sort((a, b) => a.post.mean - b.post.mean);
  const chosen = choose === 'planner' ? sorted.slice(0, racksNeeded) : racks.slice(0, racksNeeded);
  const Lday = chosen.reduce((s, r) => s + r.post.mean, 0), L = Lday / 24;
  const tau = baseline ? baseline.tau : optimalTau(C, R, L);
  const rng = mulberry32(seed), gs = [], ints = [];
  for (let i = 0; i < sims; i++) {
    let l = 0; for (const r of chosen) l += gammaSample(rng, r.post.shape) / r.post.rate;
    gs.push(goodput(tau, C, R, l / 24)); ints.push(l * 7);
  }
  const g = goodput(tau, C, R, L), gpus = racksNeeded * gpusPerRack;
  return { chosen, Lday, tau, goodput: g, gLo: quantile(gs, 0.1), gHi: quantile(gs, 0.9), perWeek: Lday * 7, perWeekLo: quantile(ints, 0.1), perWeekHi: quantile(ints, 0.9), lostGpuHoursWeek: (1 - g) * gpus * 168, mtbfHours: 24 / Lday };
}

// Replay: job runs on `racks` from t0 for `hours`, checkpoint every tau (+C), restart R after a failure,
// failures are the realized interruption times (hours) of nodes in those racks. Returns realized goodput.
export function simulateRun(failureTimes, t0, hours, tau, C, R) {
  const fails = failureTimes.filter(t => t >= t0 && t < t0 + hours).sort((a, b) => a - b);
  let t = t0, useful = 0, fi = 0; const end = t0 + hours;
  while (t < end) {
    const segEnd = t + tau + C, nextFail = fi < fails.length ? fails[fi] : Infinity;
    if (nextFail < Math.min(segEnd, end)) {           // lose the in-flight segment, restart
      t = nextFail + R; fi++;
      while (fi < fails.length && fails[fi] < t) fi++; // failures during restart are absorbed
    } else if (segEnd <= end) { useful += tau; t = segEnd; }
    else { useful += Math.max(0, Math.min(tau, end - t)); t = end; }
  }
  return useful / hours;
}
