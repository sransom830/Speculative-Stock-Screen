-- Layer 2: current speculative population + append-only history (replay).
-- Safe to run if tables already exist with compatible columns.

CREATE TABLE IF NOT EXISTS public.layer2_speculative_population (
    symbol text NOT NULL,
    speculative_attention_score numeric NOT NULL,
    attention_state text NOT NULL,
    source_cohorts jsonb NOT NULL DEFAULT '[]'::jsonb,
    relative_volume_factor numeric,
    volatility_factor numeric,
    acceleration_factor numeric,
    ecosystem_multiplier numeric,
    last_updated timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT layer2_speculative_population_pkey PRIMARY KEY (symbol)
);

COMMENT ON TABLE public.layer2_speculative_population IS
    'Layer 2 latest market-native speculative attention snapshot per symbol.';

CREATE INDEX IF NOT EXISTS idx_layer2_population_last_updated
    ON public.layer2_speculative_population (last_updated DESC);

CREATE TABLE IF NOT EXISTS public.layer2_speculative_population_history (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    snapshot_at timestamptz NOT NULL,
    symbol text NOT NULL,
    speculative_attention_score numeric NOT NULL,
    attention_state text NOT NULL,
    source_cohorts jsonb NOT NULL DEFAULT '[]'::jsonb,
    relative_volume_factor numeric,
    volatility_factor numeric,
    acceleration_factor numeric,
    ecosystem_multiplier numeric,
    CONSTRAINT layer2_speculative_population_history_pkey PRIMARY KEY (id)
);

COMMENT ON TABLE public.layer2_speculative_population_history IS
    'Append-only speculative attention snapshots for replay / reconstruction.';

CREATE INDEX IF NOT EXISTS idx_layer2_pop_hist_snapshot
    ON public.layer2_speculative_population_history (snapshot_at DESC);

CREATE INDEX IF NOT EXISTS idx_layer2_pop_hist_symbol
    ON public.layer2_speculative_population_history (symbol);
