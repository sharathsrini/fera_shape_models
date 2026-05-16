"""Build source-backed curve anomaly labels for benchmarking.

The labels are intentionally conservative:
- "gold" labels are high-confidence, source-backed market shocks in their core windows.
- "silver" labels are event shoulders or plausible recent/source-backed shocks.
- "review" labels are useful analyst-review cases but should not be treated as hard truth.
- "exclude" labels are pure data-quality/calendar carry-forward rows.

Outputs:
    labels/event_catalog.csv
    labels/curve_labels.csv
    labels/ml_wide_labelled.csv
    labels/ml_long_labelled.csv
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TENORS = [f"M{i}" for i in range(1, 37)]
MARKETS = ("TTF", "THE", "JKM")


EVENTS = [
    {
        "event_id": "JKM_2021_01_ASIA_LNG_PROMPT_SPIKE",
        "event_name": "Asia LNG cold-snap prompt-month spike",
        "markets": "JKM",
        "start_date": "2021-01-07",
        "end_date": "2021-01-29",
        "core_start": "2021-01-11",
        "core_end": "2021-01-18",
        "anomaly_family": "prompt_month_spike",
        "market_scope": "asia_lng_market_specific",
        "is_cross_market_event": False,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 3,
        "severity_shoulder": 2,
        "affected_tenors": "M1-M3",
        "shape_signature": "Extreme M1/M2 premium versus rest of JKM curve.",
        "source_urls": "https://www.iea.org/commentaries/asias-record-gas-prices-underline-the-need-to-make-its-markets-more-resilient",
        "label_note": "Cold weather, tight supply and LNG logistics drove record Asian LNG prices in Jan 2021.",
    },
    {
        "event_id": "EU_2021_10_GAS_CRUNCH_RECORD_TTF",
        "event_name": "October 2021 European gas crunch",
        "markets": "TTF|THE",
        "start_date": "2021-10-01",
        "end_date": "2021-10-07",
        "core_start": "2021-10-05",
        "core_end": "2021-10-06",
        "anomaly_family": "front_curve_supply_panic",
        "market_scope": "europe_regional",
        "is_cross_market_event": True,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 2,
        "severity_shoulder": 1,
        "affected_tenors": "M1-M12",
        "shape_signature": "European front curve repriced sharply on low storage and Russian-flow concerns.",
        "source_urls": "https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/101321-russia-prepared-to-increase-european-gas-supply-if-requested-putin",
        "label_note": "TTF day-ahead hit a high on Oct 5 amid low storage and supply concerns.",
    },
    {
        "event_id": "GLOBAL_2021_12_WINTER_GAS_CRISIS",
        "event_name": "December 2021 winter gas crisis",
        "markets": "TTF|THE|JKM",
        "start_date": "2021-12-01",
        "end_date": "2022-01-07",
        "core_start": "2021-12-20",
        "core_end": "2021-12-29",
        "anomaly_family": "front_winter_backwardation",
        "market_scope": "global_gas_cross_market",
        "is_cross_market_event": True,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 3,
        "severity_shoulder": 2,
        "affected_tenors": "M1-M16",
        "shape_signature": "Extreme winter/front premium with pronounced mid-curve cliff, especially M15-M16.",
        "source_urls": "https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/123021-russias-gazprom-sees-1-bcm-of-gas-in-european-storage-sites-by-end-december",
        "label_note": "Low European storage and lower-than-expected Russian supply pushed TTF to record highs.",
    },
    {
        "event_id": "GLOBAL_2022_03_RUSSIA_INVASION_GAS_SHOCK",
        "event_name": "Russia invasion gas shock",
        "markets": "TTF|THE|JKM",
        "start_date": "2022-02-24",
        "end_date": "2022-03-08",
        "core_start": "2022-03-02",
        "core_end": "2022-03-08",
        "anomaly_family": "war_supply_shock",
        "market_scope": "global_gas_cross_market",
        "is_cross_market_event": True,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 3,
        "severity_shoulder": 2,
        "affected_tenors": "M1-M13",
        "shape_signature": "War-driven front and first-year stress; common large kink around M12-M13.",
        "source_urls": "https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/030722-european-gas-power-price-spikes-ease-on-russia-sanctions-downplay",
        "label_note": "Russia's invasion triggered record European gas and LNG prices in early March 2022.",
    },
    {
        "event_id": "EU_2022_07_NORD_STREAM_20PCT_CUT",
        "event_name": "Nord Stream flow cut to roughly 20 percent",
        "markets": "TTF|THE",
        "start_date": "2022-07-21",
        "end_date": "2022-07-29",
        "core_start": "2022-07-25",
        "core_end": "2022-07-29",
        "anomaly_family": "pipeline_supply_cut",
        "market_scope": "europe_regional",
        "is_cross_market_event": True,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 2,
        "severity_shoulder": 1,
        "affected_tenors": "M1-M24",
        "shape_signature": "European curve stress after Nord Stream maintenance and renewed capacity fears.",
        "source_urls": "https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/072122-nord-stream-gas-flows-resume-after-maintenance-work-completed-prices-dip",
        "label_note": "Nord Stream returned from maintenance at reduced flows; further cuts were anticipated.",
    },
    {
        "event_id": "EU_2022_08_NORD_STREAM_STORAGE_SURGE",
        "event_name": "August 2022 European storage and Nord Stream price surge",
        "markets": "TTF|THE",
        "start_date": "2022-08-15",
        "end_date": "2022-08-31",
        "core_start": "2022-08-22",
        "core_end": "2022-08-29",
        "anomaly_family": "storage_injection_supply_shock",
        "market_scope": "europe_regional",
        "is_cross_market_event": True,
        "genuine_assessment": "yes",
        "confidence": "high",
        "severity_core": 3,
        "severity_shoulder": 2,
        "affected_tenors": "M1-M36",
        "shape_signature": "Whole European curve at crisis levels, with very high prompt and winter prices.",
        "source_urls": "https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/082222-european-gas-prices-surge-on-renewed-russian-gas-supply-uncertainty",
        "label_note": "Renewed Nord Stream uncertainty drove TTF close to EUR 300/MWh; ESMA later identified Russian supply disruption as the primary driver.",
    },
    {
        "event_id": "JKM_2022_08_GLOBAL_LNG_SPILLOVER",
        "event_name": "JKM spillover from global LNG crisis",
        "markets": "JKM",
        "start_date": "2022-08-22",
        "end_date": "2022-09-15",
        "core_start": "2022-08-26",
        "core_end": "2022-09-15",
        "anomaly_family": "global_lng_spillover",
        "market_scope": "global_lng_market",
        "is_cross_market_event": False,
        "genuine_assessment": "likely",
        "confidence": "medium",
        "severity_core": 2,
        "severity_shoulder": 1,
        "affected_tenors": "M1-M24",
        "shape_signature": "Asian LNG curve dislocation linked to Europe-Asia cargo competition.",
        "source_urls": "https://www.iea.org/reports/gas-market-lessons-from-the-2022-2023-energy-crisis/anatomy-of-a-natural-gas-crisis",
        "label_note": "Europe's stronger willingness to pay for LNG in 2022 reconfigured global LNG trade flows.",
    },
    {
        "event_id": "EU_2022_12_STORAGE_PRICE_CAP_REPRICING",
        "event_name": "Late-2022 European storage, price-cap and holiday repricing",
        "markets": "TTF|THE",
        "start_date": "2022-12-19",
        "end_date": "2022-12-30",
        "core_start": "2022-12-21",
        "core_end": "2022-12-27",
        "anomaly_family": "policy_weather_storage_repricing",
        "market_scope": "europe_regional",
        "is_cross_market_event": True,
        "genuine_assessment": "mixed",
        "confidence": "medium",
        "severity_core": 2,
        "severity_shoulder": 1,
        "affected_tenors": "M12-M30",
        "shape_signature": "Mid/far curve kink during rapid post-crisis repricing and year-end liquidity/calendar effects.",
        "source_urls": "https://www.consilium.europa.eu/en/press/press-releases/2022/12/19/council-agrees-on-temporary-mechanism-to-limit-excessive-gas-prices/pdf/",
        "label_note": "Useful review label, not a hard gold label: EU price-cap agreement and year-end repricing coincide with exact duplicate curves on Dec 27, indicating calendar/carry-forward risk.",
    },
    {
        "event_id": "GLOBAL_2026_03_LNG_STORAGE_SHOCK",
        "event_name": "March 2026 LNG shipping and storage shock",
        "markets": "TTF|THE|JKM",
        "start_date": "2026-03-03",
        "end_date": "2026-03-31",
        "core_start": "2026-03-03",
        "core_end": "2026-03-31",
        "anomaly_family": "lng_shipping_storage_shock",
        "market_scope": "global_lng_cross_market",
        "is_cross_market_event": True,
        "genuine_assessment": "likely",
        "confidence": "medium",
        "severity_core": 2,
        "severity_shoulder": 2,
        "affected_tenors": "M1-M25",
        "shape_signature": "Front/near-year stress with repeated kinks around M12-M13 and M24-M25 in Europe.",
        "source_urls": "https://www.gecf.org/Portals/0/xBlog/uploads/2026/3/25/GECFMonthlyGasMarketReport-March2026.pdf|https://www.spglobal.com/energy/en/news-research/latest-news/natural-gas/033126-feature-low-gas-storage-lng-disruption-to-test-european-resilience-in-q2",
        "label_note": "Recent event: source-backed but should be treated as silver until independently settlement-validated.",
    },
]


def build_time_splits(
    wide: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> pd.DataFrame:
    """Mirror ae_shape.data.time_aware_split per market."""
    parts = []
    for market, g in wide.groupby("MARKET"):
        g = g.sort_values("TRADE_DATE").copy()
        n = len(g)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        split = np.array(["train"] * n)
        split[n_train : n_train + n_val] = "val"
        split[n_train + n_val :] = "test"
        g["time_split"] = split
        parts.append(g[["TRADE_DATE", "MARKET", "time_split"]])
    return pd.concat(parts, ignore_index=True)


def tenor_number(label: str) -> int:
    match = re.fullmatch(r"M(\d+)", str(label))
    if not match:
        raise ValueError(f"Bad tenor label: {label!r}")
    return int(match.group(1))


def parse_tenor_ranges(spec: str) -> set[int]:
    out: set[int] = set()
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(tenor_number(a), tenor_number(b) + 1))
        else:
            out.add(tenor_number(part))
    return out


def event_catalog() -> pd.DataFrame:
    cat = pd.DataFrame(EVENTS)
    for col in ("start_date", "end_date", "core_start", "core_end"):
        cat[col] = pd.to_datetime(cat[col])
    return cat


def add_duplicate_flags(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.sort_values(["MARKET", "TRADE_DATE"]).copy()
    wide["is_data_quality_anomaly"] = False
    wide["data_quality_type"] = ""
    wide["data_quality_note"] = ""

    for market, idx in wide.groupby("MARKET", sort=False).groups.items():
        g = wide.loc[idx].sort_values("TRADE_DATE")
        sig = g[TENORS].round(8).astype(str).agg("|".join, axis=1)
        dup_mask = sig.eq(sig.shift())
        dup_idx = g.index[dup_mask]
        wide.loc[dup_idx, "is_data_quality_anomaly"] = True
        wide.loc[dup_idx, "data_quality_type"] = "adjacent_exact_duplicate_curve"
        wide.loc[dup_idx, "data_quality_note"] = (
            "Curve is identical to previous available curve for the same market; "
            "likely holiday/weekend/carry-forward rather than a new market shape."
        )
    return wide


def build_curve_labels(wide: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    labels = wide[["TRADE_DATE", "MARKET"]].copy()
    labels = labels.merge(build_time_splits(wide), on=["TRADE_DATE", "MARKET"], how="left")
    labels["is_market_anomaly"] = False
    labels["is_cross_market_event"] = False
    labels["event_id"] = ""
    labels["event_name"] = ""
    labels["event_phase"] = ""
    labels["anomaly_family"] = ""
    labels["market_scope"] = ""
    labels["genuine_assessment"] = "normal"
    labels["label_confidence"] = ""
    labels["severity"] = 0
    labels["affected_tenors"] = ""
    labels["shape_signature"] = ""
    labels["label_note"] = ""
    labels["source_urls"] = ""

    for _, event in catalog.iterrows():
        markets = set(str(event["markets"]).split("|"))
        mask = (
            labels["MARKET"].isin(markets)
            & labels["TRADE_DATE"].between(event["start_date"], event["end_date"])
        )
        if not mask.any():
            continue

        core_mask = mask & labels["TRADE_DATE"].between(event["core_start"], event["core_end"])
        shoulder_mask = mask & ~core_mask

        for phase_mask, phase_name, severity_col in [
            (shoulder_mask, "shoulder", "severity_shoulder"),
            (core_mask, "core", "severity_core"),
        ]:
            if not phase_mask.any():
                continue
            labels.loc[phase_mask, "is_market_anomaly"] = True
            labels.loc[phase_mask, "is_cross_market_event"] = bool(event["is_cross_market_event"])
            for col in [
                "event_id",
                "event_name",
                "anomaly_family",
                "market_scope",
                "genuine_assessment",
                "affected_tenors",
                "shape_signature",
                "label_note",
                "source_urls",
            ]:
                labels.loc[phase_mask, col] = event[col]
            labels.loc[phase_mask, "event_phase"] = phase_name
            labels.loc[phase_mask, "label_confidence"] = event["confidence"]
            labels.loc[phase_mask, "severity"] = int(event[severity_col])

    duplicate_cols = [
        "TRADE_DATE",
        "MARKET",
        "is_data_quality_anomaly",
        "data_quality_type",
        "data_quality_note",
    ]
    labels = labels.merge(wide[duplicate_cols], on=["TRADE_DATE", "MARKET"], how="left")

    labels["benchmark_tier"] = "normal"
    gold = (
        labels["is_market_anomaly"]
        & (labels["event_phase"] == "core")
        & (labels["label_confidence"] == "high")
        & (labels["genuine_assessment"] == "yes")
    )
    silver = (
        labels["is_market_anomaly"]
        & ~gold
        & labels["genuine_assessment"].isin(["yes", "likely"])
    )
    review = labels["is_market_anomaly"] & labels["genuine_assessment"].isin(["mixed"])
    exclude = labels["is_data_quality_anomaly"] & ~labels["is_market_anomaly"]

    labels.loc[gold, "benchmark_tier"] = "gold"
    labels.loc[silver, "benchmark_tier"] = "silver"
    labels.loc[review, "benchmark_tier"] = "review"
    labels.loc[exclude, "benchmark_tier"] = "exclude"
    labels.loc[
        labels["is_data_quality_anomaly"] & labels["is_market_anomaly"],
        "benchmark_tier",
    ] = labels.loc[
        labels["is_data_quality_anomaly"] & labels["is_market_anomaly"],
        "benchmark_tier",
    ].astype(str) + "_with_data_quality_flag"

    labels["benchmark_is_anomaly_strict"] = labels["benchmark_tier"].eq("gold")
    labels["benchmark_is_anomaly_broad"] = labels["benchmark_tier"].str.startswith(("gold", "silver"))

    counts = (
        labels[labels["is_market_anomaly"]]
        .groupby("TRADE_DATE")["MARKET"]
        .nunique()
        .rename("n_markets_labelled_same_date")
    )
    labels = labels.merge(counts, on="TRADE_DATE", how="left")
    labels["n_markets_labelled_same_date"] = (
        labels["n_markets_labelled_same_date"].fillna(0).astype(int)
    )
    return labels.sort_values(["TRADE_DATE", "MARKET"]).reset_index(drop=True)


def add_derived_shape_columns(wide_labelled: pd.DataFrame) -> pd.DataFrame:
    values = wide_labelled[TENORS].astype(float)
    level = values.mean(axis=1)
    wide_labelled["curve_level_mean"] = level
    wide_labelled["m1_to_m36_ratio"] = values["M1"] / values["M36"]
    wide_labelled["m1_to_level_ratio"] = values["M1"] / level
    wide_labelled["m12_to_m13_jump"] = values["M13"] - values["M12"]
    wide_labelled["m15_to_m16_jump"] = values["M16"] - values["M15"]
    wide_labelled["m24_to_m25_jump"] = values["M25"] - values["M24"]
    return wide_labelled


def add_long_tenor_flags(long_labelled: pd.DataFrame) -> pd.DataFrame:
    affected = []
    for spec, tenor in zip(long_labelled["affected_tenors"], long_labelled["TENOR_LABEL"]):
        if not isinstance(spec, str) or not spec:
            affected.append(False)
            continue
        affected.append(tenor_number(tenor) in parse_tenor_ranges(spec))
    long_labelled["is_event_affected_tenor"] = affected
    return long_labelled


def main() -> None:
    out_dir = ROOT / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(ROOT / "ml_wide.csv", parse_dates=["TRADE_DATE"])
    long = pd.read_csv(
        ROOT / "ml_long.csv",
        parse_dates=["TRADE_DATE", "START_DATE", "END_DATE", "DELIVERY_MONTH"],
    )

    catalog = event_catalog()
    wide_with_quality = add_duplicate_flags(wide)
    labels = build_curve_labels(wide_with_quality, catalog)

    wide_labelled = wide.merge(labels, on=["TRADE_DATE", "MARKET"], how="left")
    wide_labelled = add_derived_shape_columns(wide_labelled)
    long_labelled = long.merge(labels, on=["TRADE_DATE", "MARKET"], how="left")
    long_labelled = add_long_tenor_flags(long_labelled)

    catalog.to_csv(out_dir / "event_catalog.csv", index=False)
    labels.to_csv(out_dir / "curve_labels.csv", index=False)
    wide_labelled.to_csv(out_dir / "ml_wide_labelled.csv", index=False)
    long_labelled.to_csv(out_dir / "ml_long_labelled.csv", index=False)

    summary = labels.groupby(["benchmark_tier", "MARKET"]).size().rename("rows").reset_index()
    summary.to_csv(out_dir / "label_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
