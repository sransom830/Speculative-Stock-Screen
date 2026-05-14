"""
Curated symbol seeds for behavioral cohort tagging (Layer 1 observational only).

Extend these frozensets over time; membership is not a "signal" — it is provenance.
"""

from __future__ import annotations

# Retail / meme-adjacent equities (illustrative; edit freely).
RETAIL_MEME = frozenset(
    {
        "GME",
        "AMC",
        "BB",
        "NOK",
        "CLOV",
        "WISH",
        "SPCE",
        "SOFI",
        "HOOD",
        "PLTR",
        "MRNA",
        "MULN",
    }
)

# Crypto-correlated or treasury / broker / miner equities.
CRYPTO_LINKED = frozenset(
    {
        "MSTR",
        "COIN",
        "MARA",
        "RIOT",
        "CLSK",
        "CORZ",
        "IREN",
        "HUT",
        "BITF",
        "SQ",
        "PYPL",
    }
)

# AI / compute growth cluster (overlaps allowed across cohorts).
AI_GROWTH = frozenset(
    {
        "NVDA",
        "AMD",
        "SMCI",
        "PLTR",
        "PATH",
        "AI",
        "CRWD",
        "SNOW",
        "NET",
        "MDB",
        "DDOG",
        "SNPS",
        "CDNS",
        "NOW",
    }
)

# Recently listed / de-S PAC / EV / newer retail story stocks (curated, not exhaustive).
RECENT_IPO_SPAC = frozenset(
    {
        "RDDT",
        "ARM",
        "CART",
        "DASH",
        "UBER",
        "LYFT",
        "LCID",
        "RIVN",
        "RKLB",
        "ASTR",
        "SPCE",
        "OPEN",
        "DNA",
        "IONQ",
        "JOBY",
        "ACHR",
    }
)

# Names often associated with attention-driven volatility (equities only here).
VOLATILITY_ECOSYSTEMS = frozenset(
    {
        "TSLA",
        "NVDA",
        "AMD",
        "MSTR",
        "COIN",
        "GME",
        "AMC",
        "SMCI",
        "PLTR",
        "SOFI",
        "RIVN",
        "LCID",
        "NFLX",
        "SHOP",
    }
)

# Equities repeatedly discussed in short-interest / squeeze attention cycles (curated seeds).
SHORT_SQUEEZE_ECOSYSTEM = frozenset(
    {
        "CVNA",
        "GME",
        "UPST",
        "AI",
    }
)

# Speculative commercial space / aerospace equity cluster.
SPACE_AEROSPACE_SPECULATIVE = frozenset(
    {
        "RKLB",
        "ASTS",
        "LUNR",
    }
)

# High crypto-beta / treasury-as-proxy / miner names treated as levered crypto exposure (not TA).
LEVERAGED_CRYPTO_BETA = frozenset(
    {
        "MSTR",
        "COIN",
        "MARA",
        "RIOT",
    }
)

COHORT_TO_SYMBOLS: dict[str, frozenset[str]] = {
    "retail_meme": RETAIL_MEME,
    "crypto_linked": CRYPTO_LINKED,
    "ai_growth": AI_GROWTH,
    "recent_ipo_spac": RECENT_IPO_SPAC,
    "volatility_ecosystems": VOLATILITY_ECOSYSTEMS,
    "short_squeeze_ecosystem": SHORT_SQUEEZE_ECOSYSTEM,
    "space_aerospace_speculative": SPACE_AEROSPACE_SPECULATIVE,
    "leveraged_crypto_beta": LEVERAGED_CRYPTO_BETA,
}

ALL_COHORT_KEYS: frozenset[str] = frozenset(COHORT_TO_SYMBOLS.keys())
