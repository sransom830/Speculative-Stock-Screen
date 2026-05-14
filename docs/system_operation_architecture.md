# System operation architecture

This document describes the **long-term operational architecture** and **scheduling philosophy** of the speculative ecosystem stack. It is intentionally **conceptual** and **architecture-first**, while staying **implementation-aware** enough that engineers can align cron jobs, replay batches, and future live services without silent drift.

The guiding principle is **determinism in philosophy**: observables and diagnostics should be reproducible from declared inputs (universe snapshots, bars, curated narrative snapshots, replay IDs). Anything that optimizes opaque objectives or mutates behavior without traceable lineage belongs outside this core.

---

## 1. Layer separation philosophy

The system is organized as **semantically separated layers**. Each layer owns a distinct question and produces artifacts that downstream layers consume. Layers must not silently collapse into one another (for example, narrative counts must not be fused into speculative scores without an explicit contract and versioning).

### Layer 1 — Universe topology / organism discovery

Layer 1 answers: **what tradeable organisms exist in scope**, and how they relate (listing constraints, cohort tags, liquidity topology). Outputs are effectively **universe snapshots** and structural metadata used to gate who enters Layer 2.

### Layer 2 — Speculative concentration engine

Layer 2 answers: **how concentrated or active speculative participation is in market-native observables** (volume participation, volatility expansion, acceleration-shaped dynamics), conditioned on Layer 1 membership and configured scoring policies. Outputs include **speculative attention scores and states**, factor composites, and diagnostics suitable for replay validation—not execution directives.

### External narrative overlays — Narrative saturation and crowd attention topology

External overlays answer: **what does curated or adapter-supplied narrative / crowd-attention structure look like** in parallel to Layer 2 (mention velocity, breadth, headline density, thematic mass concentration). These signals are **observational enrichment**: normalized factors and convergence diagnostics relative to Layer 2 factors, **not** ingredients fused into Layer 2 scoring unless a future explicit fusion layer says otherwise.

### Regime taxonomy — Speculative ecosystem archetype classification

The regime catalog and replay tooling answer: **how do we name and compare qualitative regime shapes** (retail-led participation stacks, thematic saturation, leverage reflexivity, volatility clustering, narrative ignition) using deterministic tags and curator-linked ontology—not prediction labels. Archetypes bridge **historical presets**, **snapshots**, and **truth-set expectations** for calibration studies.

### Future Layer 3 — Fragility emergence / instability detection

Layer 3 is reserved for **fragility emergence**: combining Layer 2 concentration with overlays and taxonomy signals into explicit instability hypotheses **without** pretending they are trades. This layer does **not** exist in the core philosophy yet; keeping it unnamed in code boundaries avoids leaking “risk scoring” into Layers 1–2 prematurely.

### Future Layer 4 — Execution / confirmation / timing

Layer 4 is reserved for **execution, confirmation, and timing** policies that consume Layer 3 (when present) under strict governance. Architecture-wise it must remain **downstream** so research replay cannot be mistaken for production execution logic.

---

## 2. Operational scheduling philosophy

**Different layers run on different cadences.** Scheduling follows semantic urgency and data availability—not the convenience of a single cron expression.

| Concern | Typical cadence (examples) | Notes |
|--------|----------------------------|--------|
| Layer 1 universe | Daily / multi-day refresh | Topology shifts slowly relative to intraday tape; batch-friendly. |
| Layer 2 scoring | Intraday / hourly / future realtime | Depends on bar feed and product surface; replay validates logic offline first. |
| External narrative overlays | ~15 minutes / hourly | Bounded-ingestion cadence for normalized counts/themes; avoids tight coupling to raw vendor firehoses in core logic. |
| Replay matrix | Nightly / research batch | Deterministic batches (`replay_matrix` manifests, aggregate summaries). |
| Ontology diagnostics | Offline / weekends | Catalog compares, emission-tag bridges, taxonomy drift reviews—not latency-sensitive. |

The philosophy is **decoupled clocks**: upstream freshness does not force downstream recomputation unless contracts require it; conversely, research batches must not pretend to be realtime observers.

---

## 3. Replay philosophy

Replay infrastructure exists for:

- **Ontology refinement** — aligning named regimes and archetypes with reproducible diagnostics.
- **Archetype consistency** — stable emission tags and catalog mappings across presets and snapshots.
- **Historical saturation analysis** — Layer 2 × external overlays × convergence summaries over fixed windows.
- **Structural behavior research** — cross-regime matrices, consistency summaries, manifest lineage.

Replay infrastructure is **not** for:

- **Direct prediction optimization** — tuning objectives against hidden labels inside replay tooling.
- **Black-box ML training** — opaque model loops without declarative inputs and deterministic IDs.
- **Naive alpha mining** — harvesting correlations without structural hypotheses and versioning.

Replay outputs should remain **observable artifacts** (JSON summaries, manifests, histograms) suitable for audit and diffing.

---

## 4. Continuous improvement philosophy

The system is expected to improve through:

- **Replay accumulation** — more labeled historical bundles with stable `replay_id` lineage.
- **Ontology refinement** — tightening archetype definitions and curator links without breaking deterministic parsers.
- **Archetype expansion** — adding taxonomy entries when structural motifs recur in research.
- **Convergence analysis** — measuring alignment and divergence between market-native and narrative factors as hypotheses evolve.
- **Historical consistency** — stability diagnostics across overlapping replay dates and symbols.

Improvement is **not** framed as:

- **Uncontrolled self-modifying ML** — weights or policies mutating online without governance, versioning, and rollback tied to manifests.

Where machine learning appears in the future, it must attach to **declared datasets** (replay matrices, catalogs) and preserve **offline-first validation**.

---

## 5. Long-term architectural goal

Over time the ecosystem should converge toward:

1. **Live speculative ecosystem observer** — Layer 1–2 plus overlays on operational cadences, with explicit freshness contracts.
2. **Historical replay learner** — curated batches that accumulate structured summaries without collapsing into opaque optimization loops.
3. **Speculative saturation mapper** — comparative regimes, saturation indices, and convergence surfaces grounded in deterministic tooling.
4. **Future instability emergence detector** — Layer 3, consuming mappable signals under governance Layer 4 might eventually consume.

Each milestone preserves **traceability**: what ran, when, with which snapshots and presets.

---

## 6. Determinism and drift prevention

Operationally:

- Prefer **versioned snapshots**, **replay manifests**, and **canonical IDs** over ad hoc notebooks as sources of truth for research batches.
- Treat **catalog JSON**, **preset links**, and **truth-set events** as contracts; changing them is an ontology change, not a silent refactor.
- Keep **layer boundaries explicit** in documentation and module ownership so scheduling and ownership maps cleanly (universe vs scoring vs narrative vs taxonomy vs future fragility vs execution).

This document should be reviewed when introducing new cadences, new layers, or any fusion of narrative signals into scoring—those are architectural decisions, not incidental implementation details.
