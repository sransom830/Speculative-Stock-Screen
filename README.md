# Speculative market behavior system

Layered research stack focused on **speculative crowd behavior**, **narrative concentration**, **behavioral instability**, and **ecosystem monitoring** — not a generic equity screener, not a TA-only stack, and not a short-selling engine.

## Layers

| Layer | Question | Scope (planned) |
|-------|----------|------------------|
| 1 | What exists? | Broad liquid / speculative-capable observable universe |
| 2 | What is attracting speculative attention? | Narrative / attention (future) |
| 3 | What structures are fragile? | Instability (future) |
| 4 | Has the unwind begun? | Historical unwind cues (future) |

**Current scope:** foundational layout and **Layer 1 data plumbing** only — abstract provider interface, with room for Alpaca (or other) adapters. No execution engine, fragility logic, or signal engine.

## Repository layout

- `data/` — provider adapters (`data/providers/`), ingestion, cache, storage
- `layer1/` … `layer4/` — layer-specific domain logic (stubs for 2–4)
- `config/` — configuration
- `tests/` — tests
- `notebooks/` — exploratory work
- `scripts/` — one-off automation

See [SYSTEM_RULES.md](SYSTEM_RULES.md) for boundaries between layers and modules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Run tests from the repository root with `PYTHONPATH` including the project root, or install the project as a package when you add packaging.

```bash
PYTHONPATH=. pytest tests/
```

## Environment

Copy `.env.example` to `.env` and fill in provider credentials when you wire a concrete provider (e.g. Alpaca).
