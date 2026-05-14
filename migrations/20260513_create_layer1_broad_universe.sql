-- Layer 1: canonical observable speculative-capable ecosystem ("what exists?").
-- No scoring, fragility, or execution fields.

CREATE TABLE IF NOT EXISTS public.layer1_broad_universe (
    symbol text NOT NULL,
    company_name text,
    exchange text,
    asset_class text,
    market_cap numeric,
    avg_daily_dollar_volume numeric,
    price numeric,
    beta numeric,
    source_cohorts jsonb NOT NULL DEFAULT '[]'::jsonb,
    last_updated timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT layer1_broad_universe_pkey PRIMARY KEY (symbol)
);

COMMENT ON TABLE public.layer1_broad_universe IS
    'Layer 1 broad liquid / speculative-capable universe snapshot (observational only).';

CREATE INDEX IF NOT EXISTS idx_layer1_broad_universe_last_updated
    ON public.layer1_broad_universe (last_updated DESC);

CREATE INDEX IF NOT EXISTS idx_layer1_broad_universe_asset_class
    ON public.layer1_broad_universe (asset_class);
