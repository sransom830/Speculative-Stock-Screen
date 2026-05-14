# System rules

Rules keep the codebase modular: **one module, one responsibility**. No “god” orchestrators or mega-files.

## What this system is

- **Speculative behavior and ecosystem monitoring** across layers (attention, instability, unwind history).
- **Read-first data access** at Layer 1: observable universe and snapshots, not trading execution.

## What this system is not

- Not a generic stock screener driven only by fundamentals or static filters.
- Not a pure technical-analysis engine.
- Not a short-selling or execution product.

## Layer boundaries

1. **Layer 1** (`layer1/`, `data/providers/`, `data/ingestion/`, `data/cache/`, `data/storage/`)  
   **What exists?** Universe construction, normalization, persistence of observable facts. No “why it’s hot”, no fragility scoring, no unwind narrative.

2. **Layer 2** (`layer2/`)  
   **What is attracting speculative attention?** Narrative and crowd-facing constructs only. Must not embed execution or order lifecycle.

3. **Layer 3** (`layer3/`)  
   **What speculative structures are fragile?** Instability and structural pressure — not broker API calls or raw ingestion.

4. **Layer 4** (`layer4/`)  
   **Has the unwind historically begun?** Historical / regime comparisons — not live order placement.

## Data plane rules (`data/`)

| Area | Responsibility |
|------|----------------|
| `data/providers/` | Vendor-specific clients implementing `BaseMarketDataProvider`. Alpaca (or others) live here as small modules. |
| `data/ingestion/` | Scheduling/batch steps that call providers and emit normalized records. |
| `data/cache/` | Ephemeral or derived caches (TTL, parquet scratch, etc.). |
| `data/storage/` | Durable canonical stores and schemas. |

- **Providers** expose **read-only** market/universe operations suitable for Layer 1. They do not route orders, size positions, or emit trading signals.
- **Ingestion** may orchestrate *fetch steps* but must not own Layer 2–4 business rules.
- **No fragility or unwind logic** in `data/` except generic storage of facts.

## Config and scripts

- `config/` holds loading of settings (paths, provider selection, feature flags). No domain algorithms.
- `scripts/` are thin entrypoints; heavy logic stays in packages.
- `notebooks/` are exploratory; production paths should not depend on notebook-only code.

## Testing

- `tests/` mirrors boundaries: provider contracts, ingestion transforms, layer rules — each in small modules.

## Adding Alpaca (future)

- Subclass `BaseMarketDataProvider` in `data/providers/` (e.g. `alpaca_provider.py`).
- Map Alpaca asset JSON to `AssetRef` / `QuoteSnapshot` in that module or a dedicated `data/providers/alpaca_mapping.py` if mapping grows.
- Keep API keys and base URLs in environment variables (see `.env.example`).
