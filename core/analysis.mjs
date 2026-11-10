// Fleet-level pipeline used by the page and the tests.
import { features, standardizer, fitPoissonGLM, rateOf, rackPosterior, calibrationTable, devianceSkill, planJob, simulateRun, optimalTau } from './reliability.mjs';

const failedOn = (node, d) => node.failures.some(t => t >= d * 24 && t < (d + 1) * 24) ? 1 : 0;
const countIn = (times, d0, d1) => times.filter(t => t >= d0 * 24 && t < d1 * 24).length;

// Fit on node-days [warmup, asOf) using features from the preceding 14 days; score every node at asOf.
export function analyze(fleet, asOf, { warmup = 14, stride = 2, l2 = 2, priorDays = 60 } = {}) {
  const X = [], y = [], e = [];
  for (const n of fleet.nodes) {
    const tel = { ...n.tel, age0: n.age0 };
    for (let d = warmup; d < asOf; d += stride) {
      X.push(features(tel, d)); // features use only days before d
      let f = 0; for (let k = d; k < Math.min(asOf, d + stride); k++) f += failedOn(n, k);
      y.push(f); e.push(Math.min(stride, asOf - d));
    }
  }
  const S = standardizer(X);
  const beta = fitPoissonGLM(X.map(S.apply), y, e, { l2 });
  const nodeRate = fleet.nodes.map(n => rateOf(beta, S.apply(features({ ...n.tel, age0: n.age0 }, asOf))));
  const rackDays = fleet.racks.length * (asOf - warmup);
  const rackBase = fleet.racks.reduce((s, r) => s + countIn(r.rack_failures, warmup, asOf), 0) / rackDays;
  const racks = fleet.racks.map((r, i) => {
    const members = fleet.nodes.map((n, j) => [n, j]).filter(([n]) => n.rack === i);
    const modelRate = members.reduce((s, [, j]) => s + nodeRate[j], 0) + rackBase;
    const recent = members.reduce((s, [n]) => s + countIn(n.failures, asOf - priorDays, asOf), 0) + countIn(r.rack_failures, asOf - priorDays, asOf);
    return { id: r.id, hall: r.hall, index: i, modelRate, recent, post: rackPosterior(modelRate, recent, priorDays, priorDays), nodes: members.map(([n, j]) => ({ id: n.id, rate: nodeRate[j] })) };
  });
  return { asOf, beta, S, nodeRate, rackBase, racks, nTrain: y.length, failuresTrain: y.reduce((a, b) => a + b, 0) };
}

// Held-out reliability: node rate at asOf times window length vs failures observed in [asOf, asOf + days).
export function heldOut(fleet, A, days) {
  const pred = A.nodeRate.map(r => r * days), obs = fleet.nodes.map(n => countIn(n.failures, A.asOf, A.asOf + days));
  const mean = obs.reduce((a, b) => a + b, 0) / obs.length;
  return { table: calibrationTable(pred, obs, 5), skill: devianceSkill(pred.map(p => Math.max(p, 1e-9)), obs.map(() => Math.max(mean, 1e-9)), obs), expected: pred.reduce((a, b) => a + b, 0), observed: obs.reduce((a, b) => a + b, 0) };
}

export function failureTimesFor(fleet, rackIdx) {
  const set = new Set(rackIdx), out = [];
  fleet.nodes.forEach(n => { if (set.has(n.rack)) out.push(...n.failures); });
  fleet.racks.forEach((r, i) => { if (set.has(i)) out.push(...r.rack_failures); });
  return out.sort((a, b) => a - b);
}

// Replay a plan made at A.asOf against what actually happened afterwards.
export function backtest(fleet, A, { racksNeeded, C, R, runDays = 30, starts = 30, defaultTau = 1 }) {
  const plan = planJob(A.racks, racksNeeded, { C, R, sims: 500 });
  const naiveRacks = A.racks.slice(0, racksNeeded);
  const pick = rs => failureTimesFor(fleet, rs.map(r => r.index));
  const fp = pick(plan.chosen), fd = pick(naiveRacks);
  const runs = { planner: [], racksOnly: [], tauOnly: [], baseline: [] };
  for (let s = 0; s < starts; s++) {
    const t0 = (A.asOf + s) * 24, H = runDays * 24;
    runs.planner.push(simulateRun(fp, t0, H, plan.tau, C, R));
    runs.racksOnly.push(simulateRun(fp, t0, H, defaultTau, C, R));
    runs.tauOnly.push(simulateRun(fd, t0, H, optimalTau(C, R, naiveRacks.reduce((a, r) => a + r.post.mean, 0) / 24), C, R));
    runs.baseline.push(simulateRun(fd, t0, H, defaultTau, C, R));
  }
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
  const realizedInts = (rs, d0, d1) => countIn(pick(rs), d0, d1);
  return { plan, runs, mean: Object.fromEntries(Object.entries(runs).map(([k, v]) => [k, mean(v)])),
    interruptions: { planner: realizedInts(plan.chosen, A.asOf, A.asOf + runDays + starts), baseline: realizedInts(naiveRacks, A.asOf, A.asOf + runDays + starts) },
    windowDays: runDays + starts };
}
