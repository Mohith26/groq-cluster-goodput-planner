# Goodput Planner for GroqCloud GPU clusters (concept demo)

**Live demo:** https://mohith26.github.io/groq-cluster-goodput-planner/

Unofficial concept demo by Mohith Gajjela. Not affiliated with Groq. All fleet data is synthetic.

Groq now rents NVIDIA clusters (NVIDIA Cloud Partner since Aug 2026, GB300/Vera Rubin NVL72 racks) next to its inference cloud. For a customer training on a reserved cluster, the number that matters is **goodput**: the share of GPU time that turns into kept training progress after checkpoints, lost work and restarts. On rack-scale NVL72 systems a single tray fault stops all 72 GPUs in the NVLink domain, so which racks a job lands on, and how often it checkpoints, decide how much of the reservation is wasted.

This demo adds a **Goodput planner** tab to a new *Clusters* page in the GroqCloud console:

- **Rack interruption risk map**: 40 racks, each split into its 18 compute trays, colored by a calibrated forecast of interruptions per week. The planner keeps the lowest-risk racks for the job and holds the rest as spares.
- **Checkpoint interval**: goodput vs. checkpoint interval for the selected racks, with an 80% band from rate uncertainty, the optimum, and the customer's current interval.
- **KPIs**: expected interruptions per week (with an 80% range), projected goodput, recommended checkpoint interval, and GPU-hours recovered vs. a default allocation with hourly checkpoints.
- **Rack drill-down**: the riskiest trays in a rack and which telemetry drives each one.
- **Backtest**: the same planner run on Sep 1 and replayed against what actually failed afterwards.
- **Calibration**: forecast risk bands vs. observed interruptions in the next 60 days, and the strongest signals.

## How it works

`core/reliability.mjs` and `core/analysis.mjs`, no dependencies:

1. **Features** per tray from the previous 14 days only: correctable ECC (SBE) rate, NVLink CRC errors, non-fatal XID events, PCIe AER errors, GPU temperature above 75°C, and an early-life (burn-in) term.
2. **Poisson regression** (log link, exposure in days, L2, Newton/IRLS) of tray interruptions on those features, fit on tray-days before the planning date.
3. **Rack posterior**: an NVL72 rack's rate is the sum of its trays plus a pooled rack-level rate (switch trays, cooling, power). That model rate is a Gamma prior worth 60 days of observation, updated with the rack's own interruptions in the last 60 days.
4. **Goodput model** (first-order Young/Daly): with compute interval τ, checkpoint write C, restart R and interruption rate Λ, goodput ≈ τ/(τ+C) · (1 − Λ(R + (τ+C)/2)). The optimal τ is found by golden-section search (it matches Young's √(2C/Λ) when C is small; tested).
5. **Uncertainty**: Gamma posterior draws of each rack's rate give the 80% ranges for interruptions/week and goodput.
6. **Backtest**: fit on data up to Sep 1, choose racks and τ, then replay 30 thirty-day runs (one starting each day Sep 1–30) against the realized interruption times with an event-driven simulator (lose the in-flight interval, pay the restart), compared with racks by ID plus 1-hour checkpoints, and with each change alone.

Results on this synthetic fleet (32 racks of 40, 4-minute checkpoints, 25-minute restarts): realized goodput 89.3% → 90.9% in the backtest (rack choice and interval each contribute about half), ~27K GPU-hours per 30 days on 2,304 GPUs; the kept racks had 66 interruptions over 60 days vs 77 on the default racks. Forecast risk bands separate out of sample (lowest band 2 observed vs highest 42 over 60 days; fleet total 80 expected vs 91 observed).

## Data

`scripts/generate.mjs` (seed 20261110) builds 40 GB300 NVL72 racks × 18 trays × 4 GPUs = 2,880 GPUs over 180 days (May 4 – Oct 30, 2026). The interruption rate is calibrated to the only large public number I know of, the Llama 3 paper (Meta, 2024): 419 unexpected interruptions in 54 days on 16,384 H100s, about 78% hardware-related. Hidden tray frailty and rack effects (5 "bad batch" racks, cooling-loop temperature) drive both the hazard and noisy telemetry precursors. The truth fields are used only by tests.

## Tests

```
node --test test/reliability.test.mjs   # 7 tests
```
GLM coefficient recovery; Young vs numeric optimum and unimodality; event simulator vs analytic goodput on Poisson failures; gamma sampler moments; fleet rate vs the public calibration number; out-of-sample ranking (Spearman with hidden frailty > 0.5), deviance skill and risk-band separation; backtest planner beats default.

## Honest limitations

- **Synthetic fleet.** The telemetry-to-failure relationships are assumptions. With Groq's own GroqMetal telemetry (DCGM/XID streams, NVLink and switch counters, RMA history) the same code would fit real coefficients, and the backtest would become a real claim.
- First-order goodput model: it ignores failures during checkpoint writes and multiple failures inside one restart window (the simulator does model the latter). Asynchronous checkpointing changes C.
- Rack selection assumes the job needs whole NVLink domains and that spare racks are interchangeable; topology (rail/leaf placement) is not modeled.
- The Poisson model treats interruptions as independent given features; correlated events (a shared power shelf or switch) are only captured through the rack-level rate and the rack's own history.
- There is no public screenshot of a Groq cluster console, so the page reuses the GroqCloud console design (Space Grotesk/Inter, #F55036, neutral scale, Dashboard sidebar, Metrics/Usage card and chart styles) and places the feature as a new Clusters entry in the Dashboard sidebar.

## Novelty note

Groq's changelog, docs and newsroom (through Nov 2026) cover the inference API, Batch/Flex/Performance tiers, Projects, Spend Limits and Prometheus metrics; nothing on per-rack interruption forecasts or checkpoint planning for GPU clusters. Competitors ship parts of the area: CoreWeave Mission Control does health checks, proactive node replacement and GPU straggler detection and advertises 96% goodput. I did not find a customer-facing planner that turns calibrated rack risk into rack selection and a checkpoint interval with uncertainty.
