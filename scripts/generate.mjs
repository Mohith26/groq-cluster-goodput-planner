// Synthetic fleet: 40 GB300 NVL72 racks x 18 compute trays (4 GPUs each) = 720 nodes, 2,880 GPUs, 180 days (May 4 - Oct 30, 2026).
// Calibrated to public numbers, not to any Groq data:
//   * Llama 3 paper (Meta, 2024, sec. 3.3.4): 419 unexpected interruptions in 54 days on 16,384 H100s
//     -> 4.7e-4 interruptions per GPU-day; ~78% hardware-attributed; GPU issues 58.7%.
//   * A 4-GPU tray therefore sees ~1.9e-3 interruptions/day; rack-level causes (switch trays, cooling loop,
//     power shelf) add a small per-rack rate, and in NVL72 any tray failure stops the 72-GPU NVLink domain.
// Truth model (hidden from the planner, saved for tests only):
//   node frailty z ~ N(0, 0.65), rack effect r ~ N(0, 0.35), 5 "bad batch" racks get +0.7
//   hazard/day = base * exp(z + r + 0.04 * max(0, temp_excess)) ; telemetry are noisy precursors of z:
//   SBE ECC ~ Pois(exp(0.4 + 0.9 z)), NVLink CRC ~ Pois(exp(-0.3 + 0.6 z + 0.5 r)), non-fatal XID ~ Pois(0.04 exp(1.1 z)),
//   PCIe AER ~ Pois(0.3 exp(0.3 z)), GPU temp excess over 75C ~ N(2 + 2.5 r, 1.5)+, plus infant mortality in first 20 days.
// Usage: node scripts/generate.mjs  (seed 20261110) -> data/fleet.json
import { writeFileSync } from 'node:fs';
import { mulberry32, normalSampler } from '../core/reliability.mjs';

const SEED = 20261110, RACKS = 40, TRAYS = 18, DAYS = 180;
const rng = mulberry32(SEED), N = normalSampler(rng);
const pois = lam => { if (lam > 30) return Math.max(0, Math.round(lam + Math.sqrt(lam) * N())); let L = Math.exp(-lam), k = 0, p = 1; do { k++; p *= rng(); } while (p > L); return k - 1; };
const PER_GPU_DAY = 419 / (16384 * 54); // Llama 3
const NODE_BASE = 4 * PER_GPU_DAY * 0.78 / Math.exp(0.65 ** 2 / 2 + 0.35 ** 2 / 2); // hardware share; mean-corrected for frailty
const RACK_BASE = 4 * PER_GPU_DAY * 0.22 * TRAYS / 6; // non-tray causes (switch, cooling, power), per rack-day
const badRacks = new Set([5, 14, 19, 27, 33]);
const racks = [], nodes = [];
for (let r = 0; r < RACKS; r++) {
  const reff = 0.35 * N() + (badRacks.has(r) ? 0.7 : 0);
  const rackFail = [];
  for (let d = 0; d < DAYS; d++) if (rng() < 1 - Math.exp(-RACK_BASE * Math.exp(0.5 * reff))) rackFail.push(+(d * 24 + rng() * 24).toFixed(3));
  racks.push({ id: `R${String(r + 1).padStart(2, '0')}`, hall: r < 12 ? 'A' : 'B', rack_effect: +reff.toFixed(4), rack_failures: rackFail });
  const age0 = Math.floor(rng() * 40); // racks brought up in waves
  for (let t = 0; t < TRAYS; t++) {
    const z = 0.65 * N();
    const tel = { sbe_ecc: [], nvlink_crc: [], xid_warn: [], pcie_aer: [], temp_excess: [] }, fails = [];
    for (let d = 0; d < DAYS; d++) {
      const temp = Math.max(0, 2 + 2.5 * reff + 1.5 * N());
      tel.sbe_ecc.push(pois(Math.exp(0.4 + 0.9 * z)));
      tel.nvlink_crc.push(pois(Math.exp(-0.3 + 0.6 * z + 0.5 * reff)));
      tel.xid_warn.push(pois(0.04 * Math.exp(1.1 * z)));
      tel.pcie_aer.push(pois(0.3 * Math.exp(0.3 * z)));
      tel.temp_excess.push(+temp.toFixed(2));
      const infant = (age0 + d) < 20 ? 2.0 : 1;
      const h = NODE_BASE * infant * Math.exp(z + reff + 0.04 * temp);
      if (rng() < 1 - Math.exp(-h)) fails.push(+(d * 24 + rng() * 24).toFixed(3));
    }
    nodes.push({ id: `R${String(r + 1).padStart(2, '0')}-T${String(t + 1).padStart(2, '0')}`, rack: r, age0, tel, failures: fails, truth_z: +z.toFixed(4) });
  }
}
writeFileSync(new URL('../data/fleet.json', import.meta.url), JSON.stringify({
  seed: SEED, days: DAYS, gpus_per_node: 4, nodes_per_rack: TRAYS,
  note: 'SYNTHETIC fleet telemetry and interruptions, calibrated to the Llama 3 paper interruption rate (419 unexpected interruptions / 54 days / 16,384 GPUs). truth_z and rack_effect are hidden from the planner and used only by tests.',
  per_gpu_day_rate_public: PER_GPU_DAY, racks, nodes }));
const nf = nodes.reduce((s, n) => s + n.failures.length, 0), rf = racks.reduce((s, r) => s + r.rack_failures.length, 0);
console.log(`${nodes.length} nodes, ${nf} node interruptions + ${rf} rack-level over ${DAYS} days; per-GPU-day ${((nf + rf) / (nodes.length * 4 * DAYS)).toExponential(2)} (public ${PER_GPU_DAY.toExponential(2)})`);
