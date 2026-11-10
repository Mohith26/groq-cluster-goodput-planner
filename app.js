import * as RL from './core/reliability.mjs';
import { analyze, heldOut, backtest } from './core/analysis.mjs';

const $ = s => document.querySelector(s);
const DAY0 = Date.UTC(2026, 4, 4); // fleet day 0 = May 4, 2026
const dstr = d => new Date(DAY0 + d * 864e5).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const hm = h => { const m = Math.round(h * 60); return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`; };
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const k1 = x => x >= 1000 ? `${(x / 1000).toFixed(1)}K` : Math.round(x).toLocaleString('en-US');

const S = { n: 32, C: 4, R: 25, cur: 60, sel: null, spares: true };
let F, NOW, PAST, HO, BT;

async function main() {
  F = await (await fetch('data/fleet.json')).json();
  NOW = analyze(F, F.days);          // plan for a job starting now
  PAST = analyze(F, F.days - 60);    // the same planner run 60 days ago, for the backtest
  HO = heldOut(F, PAST, 60);
  $('#fleet-size').textContent = `${F.racks.length} racks`;
  $('#asof').textContent = `${dstr(F.days)}, 2026`;
  $('#bt-asof').textContent = dstr(PAST.asOf);
  for (const [id, key] of [['#in-n', 'n'], ['#in-c', 'C'], ['#in-r', 'R']]) $(id).addEventListener('input', e => { S[key] = +e.target.value; render(); });
  $('#spares').addEventListener('click', () => { S.spares = !S.spares; render(); });
  render();
}

const riskColor = (v, max) => { // white -> amber -> groq orange
  const t = Math.min(1, v / max);
  const a = [254, 243, 199], b = [251, 146, 60], c = [245, 80, 54];
  const m = t < 0.5 ? a.map((x, i) => x + (b[i] - x) * t * 2) : b.map((x, i) => x + (c[i] - x) * (t - 0.5) * 2);
  return `rgb(${m.map(Math.round).join(',')})`;
};

function render() {
  const C = S.C / 60, R = S.R / 60, cur = S.cur / 60;
  const plan = RL.planJob(NOW.racks, S.n, { C, R, sims: 1500, seed: 3 });
  const naive = RL.planJob(NOW.racks, S.n, { C, R, sims: 10, choose: 'first', baseline: { tau: cur } });
  const gpus = S.n * 72, month = 720;
  const gain = (plan.goodput - naive.goodput) * gpus * month;
  $('#v-n').textContent = `${S.n} racks · ${gpus.toLocaleString('en-US')} GPUs`;
  $('#v-c').textContent = `${S.C} min`;
  $('#v-r').textContent = `${S.R} min`;

  $('#k-int').innerHTML = `${plan.perWeek.toFixed(1)}<small>/ week</small>`;
  $('#k-int-d').innerHTML = `80% range <b>${plan.perWeekLo.toFixed(1)}–${plan.perWeekHi.toFixed(1)}</b> on the ${S.n} selected racks · mean time between interruptions <b>${plan.mtbfHours.toFixed(1)} h</b>`;
  $('#k-good').innerHTML = pct(plan.goodput);
  $('#k-good-d').innerHTML = `80% range <b>${pct(plan.gLo)}–${pct(plan.gHi)}</b> · ${pct(1 - plan.goodput)} of GPU time lost to checkpoints, rework and restarts`;
  $('#k-tau').innerHTML = hm(plan.tau);
  const gCur = RL.goodput(cur, C, R, plan.Lday / 24);
  $('#k-tau-d').innerHTML = `vs. your current ${hm(cur)}: <b class="up">+${((plan.goodput - gCur) * 100).toFixed(2)} pp</b> goodput on the same racks (optimum for ${S.C} min writes, ${S.R} min restarts)`;
  $('#k-gpuh').innerHTML = `${gain >= 0 ? '+' : ''}${k1(gain)}<small>/ 30 days</small>`;
  $('#k-gpuh-d').innerHTML = `Projected vs. racks ${NOW.racks[0].id}–${NOW.racks[S.n - 1].id} with ${hm(cur)} checkpoints (<b>${pct(naive.goodput)}</b>). Realized on history: see backtest.`;

  drawRacks(plan);
  drawCurve(plan, C, R, cur);
  drawRack(plan);
  drawBacktest(C, R, cur);
  drawCalibration();
  if (window.lucide) window.lucide.createIcons();
}

function drawRacks(plan) {
  const chosen = new Set(plan.chosen.map(r => r.index));
  const wk = NOW.racks.map(r => r.post.mean * 7), max = Math.max(...wk);
  const nodeMax = Math.max(...NOW.nodeRate) * 30;
  if (S.sel === null || !NOW.racks[S.sel]) S.sel = [...NOW.racks].sort((a, b) => b.post.mean - a.post.mean).find(r => chosen.has(r.index))?.index ?? 0;
  $('#map-cap').textContent = `Bayesian rate per rack: node model fit on ${NOW.nTrain.toLocaleString('en-US')} tray-days of telemetry, updated with each rack's last 60 days. Planner keeps the ${S.n} lowest-risk racks.`;
  $('#racks').innerHTML = NOW.racks.map((r, i) => {
    const isC = chosen.has(i);
    if (!S.spares && !isC) return '';
    const cells = r.nodes.map(n => `<i style="background:${riskColor(n.rate * 30, nodeMax)}"></i>`).join('');
    return `<div class="rack ${isC ? 'chosen' : 'spare'} ${i === S.sel ? 'sel' : ''}" data-i="${i}"><div class="id"><span>${r.id}</span><span>${r.hall}</span></div><div class="bars">${cells}</div><div class="r">${wk[i].toFixed(2)}<span style="color:#a5a09c">/wk</span></div></div>`;
  }).join('');
  document.querySelectorAll('.rack').forEach(el => el.addEventListener('click', () => { S.sel = +el.dataset.i; render(); }));
  $('#scale').innerHTML = [0, .25, .5, .75, 1].map(t => `<i style="background:${riskColor(t * max, max)}"></i>`).join('');
  $('#scale-max').textContent = `0 – ${max.toFixed(2)}`;
}

function drawCurve(plan, C, R, cur) {
  const W = Math.max(360, Math.round($('#curve').clientWidth || 600)), H = 232, l = 40, r = 14, t = 14, b = 26;
  const taus = Array.from({ length: 80 }, (_, i) => 20 / 60 + i * (6 - 20 / 60) / 79);
  const rng = RL.mulberry32(9), lam = [];
  for (let s = 0; s < 400; s++) { let x = 0; for (const rk of plan.chosen) x += RL.gammaSample(rng, rk.post.shape) / rk.post.rate; lam.push(x / 24); }
  const mid = taus.map(x => RL.goodput(x, C, R, plan.Lday / 24));
  const lo = taus.map(x => RL.quantile(lam.map(L => RL.goodput(x, C, R, L)), 0.1));
  const hi = taus.map(x => RL.quantile(lam.map(L => RL.goodput(x, C, R, L)), 0.9));
  const ymin = Math.floor(Math.min(...lo) * 50) / 50, ymax = Math.min(1, Math.ceil((Math.max(...hi) + 0.005) * 50) / 50);
  const step = (ymax - ymin) > 0.12 ? 0.04 : 0.02;
  const X = v => l + (W - l - r) * (v - taus[0]) / (taus[taus.length - 1] - taus[0]), Y = v => t + (H - t - b) * (1 - (v - ymin) / (ymax - ymin));
  const path = arr => arr.map((v, i) => `${i ? 'L' : 'M'}${X(taus[i]).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  const band = path(hi) + taus.slice().reverse().map((x, i) => `L${X(x).toFixed(1)},${Y(lo[lo.length - 1 - i]).toFixed(1)}`).join('') + 'Z';
  const yt = []; for (let v = ymin; v <= ymax + 1e-9; v += step) yt.push(v);
  const gCur = RL.goodput(cur, C, R, plan.Lday / 24);
  $('#curve-cap').textContent = `Expected goodput on the selected racks (line) with 80% band from rate uncertainty. Each interruption costs the in-flight interval plus a ${S.R} min restart.`;
  $('#curve').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
    <line x1="${l}" x2="${l}" y1="${t}" y2="${H - b}" stroke="#d6d3d1"/><line x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}" stroke="#d6d3d1"/>
    ${yt.map(v => `<line x1="${l - 4}" x2="${l}" y1="${Y(v)}" y2="${Y(v)}" stroke="#d6d3d1"/><text class="axis" x="${l - 7}" y="${Y(v) + 3}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`).join('')}
    ${[0.5, 1, 2, 3, 4, 5, 6].map(v => `<line x1="${X(v)}" x2="${X(v)}" y1="${H - b}" y2="${H - b + 4}" stroke="#d6d3d1"/><text class="axis" x="${X(v)}" y="${H - 8}" text-anchor="middle">${v < 1 ? '30m' : v + 'h'}</text>`).join('')}
    <path d="${band}" fill="#22c55e" opacity="0.13"/>
    <path d="${path(mid)}" fill="none" stroke="#22c55e" stroke-width="1.6"/>
    <line x1="${X(cur)}" x2="${X(cur)}" y1="${Y(gCur)}" y2="${H - b}" stroke="#a5a09c" stroke-dasharray="3 3"/>
    <circle cx="${X(cur)}" cy="${Y(gCur)}" r="3.5" fill="#fff" stroke="#766f6b" stroke-width="1.5"/>
    <text class="axis" x="${X(cur) + 6}" y="${H - b - 6}">current ${hm(cur)} · ${pct(gCur)}</text>
    <line x1="${X(plan.tau)}" x2="${X(plan.tau)}" y1="${Y(plan.goodput)}" y2="${H - b}" stroke="#f55036" stroke-dasharray="3 3"/>
    <circle cx="${X(plan.tau)}" cy="${Y(plan.goodput)}" r="4" fill="#f55036"/>
    <text class="axis" x="${X(plan.tau) + 7}" y="${Y(plan.goodput) - 7}" style="fill:#0c0a09">optimum ${hm(plan.tau)} · ${pct(plan.goodput)}</text>
  </svg>`;
}

function drawRack(plan) {
  const r = NOW.racks[S.sel], chosen = plan.chosen.some(x => x.index === S.sel);
  const beta = NOW.beta, Sd = NOW.S;
  $('#rack-title').innerHTML = `${r.id} · Hall ${r.hall} <span class="pill ${chosen ? '' : 'hot'}">${chosen ? 'in job' : 'spare'}</span>`;
  $('#rack-cap').textContent = `Model ${(r.modelRate * 7).toFixed(2)}/wk, observed ${r.recent} in the last 60 days → posterior ${(r.post.mean * 7).toFixed(2)}/wk. In NVL72 any tray fault stops all 72 GPUs, so the rack rate is the sum of its trays.`;
  const rows = r.nodes.map(n => {
    const node = F.nodes.find(x => x.id === n.id), tel = { ...node.tel, age0: node.age0 };
    const x = RL.features(tel, F.days), z = Sd.apply(x);
    const contrib = RL.FEATURES.map((f, j) => [f, beta[j + 1] * z[j]]).sort((a, b) => b[1] - a[1])[0];
    const mean = k => { let s = 0; for (let d = F.days - 14; d < F.days; d++) s += node.tel[k][d]; return s / 14; };
    return { id: n.id.split('-')[1], exp: n.rate * 30, sbe: mean('sbe_ecc'), crc: mean('nvlink_crc'), temp: mean('temp_excess'), drv: contrib[1] > 0.05 ? RL.FEATURE_LABEL[contrib[0]].split(' (')[0].replace(' / day', '') : 'none (baseline)' };
  }).sort((a, b) => b.exp - a.exp).slice(0, 6);
  $('#trays').innerHTML = rows.map(x => `<tr><td class="mono">${x.id}</td><td class="n">${x.exp.toFixed(2)}</td><td class="n">${x.sbe.toFixed(1)}</td><td class="n">${x.crc.toFixed(1)}</td><td class="n">+${x.temp.toFixed(1)}°</td><td>${x.drv}</td></tr>`).join('');
}

function drawBacktest(C, R, cur) {
  BT = backtest(F, PAST, { racksNeeded: S.n, C, R, runDays: 30, starts: 30, defaultTau: cur });
  const rows = [['Racks by ID, ' + hm(cur) + ' checkpoints', BT.mean.baseline], ['Rack selection only', BT.mean.racksOnly], ['Interval only', BT.mean.tauOnly], ['Planner (both)', BT.mean.planner]];
  const lo = Math.min(...rows.map(x => x[1])) - 0.02, hi = Math.max(...rows.map(x => x[1])) + 0.005;
  $('#bt-cap').textContent = `Planner fit on data up to ${dstr(PAST.asOf)}, then replayed against what actually failed: 30 runs of 30 days, one starting each day ${dstr(PAST.asOf)} – ${dstr(PAST.asOf + 29)}.`;
  $('#bt-rows').innerHTML = rows.map(([nm, v], i) => `<div class="bt-row"><span>${nm}</span><span class="track"><i class="${i === 3 ? 'best' : ''}" style="width:${(100 * (v - lo) / (hi - lo)).toFixed(1)}%"></i></span><span class="val">${pct(v)}</span></div>`).join('');
  const gpuh = (BT.mean.planner - BT.mean.baseline) * S.n * 72 * 720;
  $('#bt-call').innerHTML = `Realized goodput <b>${pct(BT.mean.baseline)} → ${pct(BT.mean.planner)}</b>, about <b>${k1(gpuh)} GPU-hours per 30 days</b> on ${(S.n * 72).toLocaleString('en-US')} GPUs. The racks it kept had <b>${BT.interruptions.planner}</b> interruptions over ${BT.windowDays} days vs. <b>${BT.interruptions.baseline}</b> on the default racks.`;
}

function drawCalibration() {
  const W = Math.max(300, Math.round($('#cal').clientWidth || 380)), H = 150, l = 30, r = 8, t = 8, b = 22;
  const tb = HO.table, ymax = Math.ceil(Math.max(...tb.map(x => Math.max(x.hi, x.expected))) / 10) * 10;
  const bw = (W - l - r) / tb.length, Y = v => t + (H - t - b) * (1 - v / ymax);
  $('#cal-cap').textContent = `Trays grouped into 5 risk bands by the ${dstr(PAST.asOf)} forecast; expected vs. observed interruptions in the next 60 days (bars: observed with 90% Poisson range).`;
  $('#cal').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
    <line x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}" stroke="#d6d3d1"/>
    ${[0, ymax / 2, ymax].map(v => `<text class="axis" x="${l - 5}" y="${Y(v) + 3}" text-anchor="end">${v}</text><line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#f3f3f2"/>`).join('')}
    ${tb.map((x, i) => { const cx = l + bw * i + bw / 2; return `<rect x="${cx - bw * 0.2}" y="${Y(x.observed)}" width="${bw * 0.4}" height="${H - b - Y(x.observed)}" fill="#22c55e" rx="2"/>
      <line x1="${cx}" x2="${cx}" y1="${Y(x.hi)}" y2="${Y(x.lo)}" stroke="#15803d"/>
      <line x1="${cx - bw * 0.3}" x2="${cx + bw * 0.3}" y1="${Y(x.expected)}" y2="${Y(x.expected)}" stroke="#0c0a09" stroke-width="2"/>
      <text class="axis" x="${cx}" y="${H - 8}" text-anchor="middle">${['lowest', '2', '3', '4', 'highest'][i]}</text>`; }).join('')}
  </svg>`;
  const rr = RL.FEATURES.map((f, j) => [RL.FEATURE_LABEL[f], Math.exp(PAST.beta[j + 1])]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  $('#drivers').innerHTML = `<thead><tr><th>Strongest signals (rate ratio per +1 SD)</th><th class="n">×</th></tr></thead><tbody>${rr.map(([n, v]) => `<tr><td>${n}</td><td class="n">${v.toFixed(2)}</td></tr>`).join('')}<tr><td>Fleet total, next 60 days</td><td class="n">${HO.expected.toFixed(0)} exp. / ${HO.observed} obs.</td></tr></tbody>`;
}

main().catch(e => { document.body.insertAdjacentHTML('beforeend', `<pre style="color:#b91c1c;padding:16px">${String(e && e.stack || e)}</pre>`); console.error(e); });
