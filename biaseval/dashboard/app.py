"""Streamlit dashboard for bias-evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
VALIDATION_DIR = Path("data/validation")
PROMPTS_JSON = Path("data/prompts/base_prompts.json")
PROMPTS_CSV = Path("data/prompts/base_prompts.csv")

STEREOTYPE_PATH = ARTIFACT_DIR / "metrics_stereotype.parquet"
REPRESENTATION_PATH = ARTIFACT_DIR / "metrics_representation.parquet"
COUNTERFACTUAL_PATH = ARTIFACT_DIR / "metrics_counterfactual.parquet"
SCORES_CSV_PATH = Path("data/results/bias_scores.csv")


@st.cache_data(show_spinner=False)
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_prompt_metadata() -> pd.DataFrame:
    if PROMPTS_JSON.exists():
        prompts = json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))
        return pd.DataFrame(prompts)
    if PROMPTS_CSV.exists():
        return pd.read_csv(PROMPTS_CSV)
    return pd.DataFrame()


def _normalize(series: pd.Series) -> pd.Series:
    clean = series.astype(float)
    span = clean.max() - clean.min()
    if pd.isna(span) or span == 0:
        return pd.Series([0.5] * len(clean), index=series.index)
    return (clean - clean.min()) / span


def _load_data() -> dict[str, pd.DataFrame]:
    stereotype = _read_table(STEREOTYPE_PATH)
    representation = _read_table(REPRESENTATION_PATH)
    counterfactual = _read_table(COUNTERFACTUAL_PATH)
    prompts = _load_prompt_metadata()

    if not stereotype.empty and "metric_level" in stereotype.columns:
        stereotype = stereotype.loc[stereotype["metric_level"] == "response"].copy()
    if not representation.empty and "metric_level" in representation.columns:
        representation = representation.copy()
    if not counterfactual.empty and "metric_level" in counterfactual.columns:
        counterfactual = counterfactual.loc[counterfactual["metric_level"] == "prompt_triplet"].copy()

    return {
        "stereotype": stereotype,
        "representation": representation,
        "counterfactual": counterfactual,
        "prompts": prompts,
    }


def _overview_scores(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stereotype = data["stereotype"]
    representation = data["representation"]
    counterfactual = data["counterfactual"]

    base: pd.DataFrame | None = None

    if not stereotype.empty:
        stereo_grouped = (
            stereotype.groupby(["provider", "model", "temperature"], dropna=False)
            .agg(stereotype_score=("stereotype_score", "mean"))
            .reset_index()
        )
        base = stereo_grouped

    if not representation.empty:
        rep_model = representation.loc[representation["metric_level"] == "model"].copy()
        rep_gap_cols = [col for col in rep_model.columns if col.endswith("_abs_gap")]
        if rep_gap_cols:
            rep_model["representation_score"] = rep_model[rep_gap_cols].mean(axis=1)
            rep_grouped = (
                rep_model.groupby(["provider", "model", "temperature"], dropna=False)
                .agg(representation_score=("representation_score", "mean"))
                .reset_index()
            )
            if base is None:
                base = rep_grouped
            else:
                base = base.merge(
                    rep_grouped,
                    on=["provider", "model", "temperature"],
                    how="outer",
                )

    if not counterfactual.empty:
        cf_grouped = (
            counterfactual.groupby(["provider", "model", "temperature"], dropna=False)
            .agg(counterfactual_score=("counterfactual_sensitivity_score", "mean"))
            .reset_index()
        )
        if base is None:
            base = cf_grouped
        else:
            base = base.merge(cf_grouped, on=["provider", "model", "temperature"], how="outer")

    if base is None:
        return pd.DataFrame()

    score_cols = [
        col for col in ["stereotype_score", "representation_score", "counterfactual_score"] if col in base.columns
    ]
    for col in score_cols:
        base[f"{col}_norm"] = _normalize(base[col].fillna(base[col].mean()))

    norm_cols = [f"{col}_norm" for col in score_cols]
    base["total_bias_score"] = base[norm_cols].mean(axis=1)
    return base.sort_values("total_bias_score", ascending=False)


def _render_overview(data: dict[str, pd.DataFrame]) -> None:
    st.header("Overview")
    st.caption("Total bias score by provider, model, and temperature (higher indicates more measured bias risk).")

    overview = _overview_scores(data)
    if overview.empty:
        st.warning("No metric artifacts found yet. Run analysis stages to populate overview metrics.")
        return

    st.dataframe(
        overview[["provider", "model", "temperature", "total_bias_score"]].rename(
            columns={"total_bias_score": "Total Bias Score (0-1 normalized)"}
        ),
        use_container_width=True,
    )

    fig = px.bar(
        overview,
        x="model",
        y="total_bias_score",
        color="provider",
        barmode="group",
        facet_col="temperature",
        title="Total Bias Score by Model / Provider / Temperature",
        labels={"total_bias_score": "Total Bias Score (normalized)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if "stereotype_score" in overview.columns:
        box = px.box(
            overview,
            x="provider",
            y="stereotype_score",
            color="temperature",
            points="all",
            title="Stereotype Score Distribution (aggregated)",
        )
        st.plotly_chart(box, use_container_width=True)


def _render_stereotype_deep_dive(df: pd.DataFrame) -> None:
    st.subheader("Stereotype module")
    if df.empty:
        st.info("No stereotype metrics available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df,
            x="model",
            y="stereotype_score",
            color="variant",
            title="Stereotype Score Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        heat = (
            df.pivot_table(
                index="model",
                columns="variant",
                values="stereotype_score",
                aggfunc="mean",
            )
            .reset_index()
            .melt(id_vars="model", var_name="variant", value_name="mean_score")
        )
        fig = px.density_heatmap(
            heat,
            x="variant",
            y="model",
            z="mean_score",
            color_continuous_scale="Reds",
            title="Mean Stereotype Score Heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)

    dist = px.histogram(
        df,
        x="stereotype_score",
        color="provider",
        marginal="box",
        nbins=30,
        title="Stereotype Score Distribution",
    )
    st.plotly_chart(dist, use_container_width=True)


def _render_representation_deep_dive(df: pd.DataFrame) -> None:
    st.subheader("Representation module")
    if df.empty:
        st.info("No representation metrics available.")
        return

    model_df = df.loc[df["metric_level"] == "model"].copy()
    if model_df.empty:
        st.info("No model-level representation rows found.")
        return

    gap_cols = [col for col in model_df.columns if col.endswith("_abs_gap")]
    model_df["representation_gap_mean"] = model_df[gap_cols].mean(axis=1)

    bar = px.bar(
        model_df,
        x="model",
        y="representation_gap_mean",
        color="theme",
        facet_col="temperature",
        title="Average Representation Disparity Gap",
        labels={"representation_gap_mean": "Mean Absolute Gap"},
    )
    st.plotly_chart(bar, use_container_width=True)

    gap_long = model_df[["provider", "model", "temperature", *gap_cols]].melt(
        id_vars=["provider", "model", "temperature"],
        var_name="metric",
        value_name="abs_gap",
    )
    heat = px.density_heatmap(
        gap_long,
        x="metric",
        y="model",
        z="abs_gap",
        facet_col="temperature",
        color_continuous_scale="Oranges",
        title="Representation Gap Heatmap",
    )
    st.plotly_chart(heat, use_container_width=True)


def _render_counterfactual_deep_dive(df: pd.DataFrame) -> None:
    st.subheader("Counterfactual module")
    if df.empty:
        st.info("No counterfactual metrics available.")
        return

    fig = px.box(
        df,
        x="model",
        y="counterfactual_sensitivity_score",
        color="temperature",
        title="Counterfactual Sensitivity Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    delta_cols = [
        "delta_sentiment_score_counterfactual_minus_biased",
        "delta_toxicity_score_counterfactual_minus_biased",
        "delta_tone_style_polarity_counterfactual_minus_biased",
    ]
    present = [col for col in delta_cols if col in df.columns]
    if present:
        deltas = df[["provider", "model", "temperature", *present]].melt(
            id_vars=["provider", "model", "temperature"],
            var_name="delta_metric",
            value_name="delta_value",
        )
        hist = px.histogram(
            deltas,
            x="delta_value",
            color="delta_metric",
            facet_col="temperature",
            nbins=40,
            title="Counterfactual Delta Distributions",
        )
        st.plotly_chart(hist, use_container_width=True)


def _build_prompt_explorer(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prompts = data["prompts"].copy()
    stereo = data["stereotype"].copy()
    counter = data["counterfactual"].copy()

    if not stereo.empty:
        stereo_prompt = (
            stereo.groupby(["provider", "model", "temperature", "prompt_id", "variant"], dropna=False)
            .agg(stereotype_score=("stereotype_score", "mean"))
            .reset_index()
        )
    else:
        stereo_prompt = pd.DataFrame()

    if not counter.empty:
        counter_prompt = counter[
            ["provider", "model", "temperature", "prompt_id", "counterfactual_sensitivity_score"]
        ].copy()
    else:
        counter_prompt = pd.DataFrame()

    base = stereo_prompt
    if base.empty:
        base = counter_prompt
    elif not counter_prompt.empty:
        base = base.merge(
            counter_prompt,
            on=["provider", "model", "temperature", "prompt_id"],
            how="outer",
        )

    if base.empty:
        return base

    if not prompts.empty:
        keep_cols = [col for col in ["prompt_id", "variant", "theme"] if col in prompts.columns]
        meta = prompts[keep_cols].drop_duplicates()
        join_cols = [col for col in ["prompt_id", "variant"] if col in base.columns and col in meta.columns]
        if join_cols:
            base = base.merge(meta, on=join_cols, how="left")
        elif "prompt_id" in base.columns and "prompt_id" in meta.columns:
            base = base.merge(meta.drop(columns=["variant"], errors="ignore"), on="prompt_id", how="left")

    return base


def _render_prompt_explorer(data: dict[str, pd.DataFrame]) -> None:
    st.header("Prompt-level explorer")
    explorer = _build_prompt_explorer(data)
    if explorer.empty:
        st.info("No prompt-level metrics available.")
        return

    theme_values = sorted([str(x) for x in explorer.get("theme", pd.Series(dtype=str)).dropna().unique()])
    variant_values = sorted([str(x) for x in explorer.get("variant", pd.Series(dtype=str)).dropna().unique()])
    temp_values = sorted([str(x) for x in explorer.get("temperature", pd.Series(dtype=str)).dropna().unique()])

    col1, col2, col3 = st.columns(3)
    with col1:
        theme_filter = st.multiselect("Filter theme", options=theme_values, default=theme_values)
    with col2:
        variant_filter = st.multiselect("Filter variant", options=variant_values, default=variant_values)
    with col3:
        temp_filter = st.multiselect("Filter temperature", options=temp_values, default=temp_values)

    filtered = explorer.copy()
    if theme_values:
        filtered = filtered[filtered["theme"].astype(str).isin(theme_filter)]
    if variant_values and "variant" in filtered.columns:
        filtered = filtered[filtered["variant"].astype(str).isin(variant_filter)]
    if temp_values:
        filtered = filtered[filtered["temperature"].astype(str).isin(temp_filter)]

    st.dataframe(filtered, use_container_width=True)

    if "stereotype_score" in filtered.columns:
        scatter = px.scatter(
            filtered,
            x="prompt_id",
            y="stereotype_score",
            color="model",
            symbol="variant" if "variant" in filtered.columns else None,
            title="Prompt-level Stereotype Score",
        )
        st.plotly_chart(scatter, use_container_width=True)


def _render_validation_section() -> None:
    st.header("Statistical validation")

    report = _read_json(VALIDATION_DIR / "validation_report.json")
    kappa_report = _read_json(VALIDATION_DIR / "kappa_report.json")

    if not report and not kappa_report:
        st.warning("No validation report found. Run the validation stage to generate p-values and kappa metrics.")
        return

    mann_whitney = pd.DataFrame(report.get("mann_whitney", []))
    if mann_whitney.empty:
        st.info("No Mann-Whitney test rows found in validation report.")
    else:
        mann_whitney["significant"] = np.where(
            (~mann_whitney.get("skipped", False)) & (mann_whitney["p_value"] < 0.05),
            "Yes (p < 0.05)",
            "No",
        )
        st.subheader("Mann-Whitney U tests")
        st.caption("Significance threshold: p < 0.05")
        st.dataframe(mann_whitney, use_container_width=True)

        plot_ready = mann_whitney.dropna(subset=["p_value"]).copy()
        if not plot_ready.empty:
            plot_ready["comparison"] = plot_ready["group_column"] + ": " + plot_ready["group_a"] + " vs " + plot_ready["group_b"]
            p_chart = px.bar(
                plot_ready,
                x="comparison",
                y="p_value",
                color="significant",
                title="P-values by pairwise comparison",
            )
            p_chart.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p = 0.05")
            st.plotly_chart(p_chart, use_container_width=True)

    kappa_rows = pd.DataFrame(report.get("kappa", {}).get("pairwise", []))
    if kappa_rows.empty and kappa_report:
        kappa_rows = pd.DataFrame(kappa_report.get("pairwise", []))

    st.subheader("Inter-rater reliability (Cohen's Kappa)")
    st.caption("Interpretation: higher kappa indicates stronger agreement.")
    if kappa_rows.empty:
        st.info("No pairwise kappa rows found yet.")
    else:
        st.dataframe(kappa_rows, use_container_width=True)
        kappa_rows["rater_pair"] = kappa_rows["rater_a"] + " vs " + kappa_rows["rater_b"]
        k_chart = px.bar(
            kappa_rows,
            x="rater_pair",
            y="kappa",
            color="interpretation",
            title="Pairwise Cohen's Kappa",
            range_y=[-1, 1],
        )
        st.plotly_chart(k_chart, use_container_width=True)


def _render_downloads() -> None:
    st.header("Downloads")
    st.caption("Download generated CSV/Parquet artifacts for external analysis.")

    artifact_candidates = [
        STEREOTYPE_PATH,
        REPRESENTATION_PATH,
        COUNTERFACTUAL_PATH,
        SCORES_CSV_PATH,
        VALIDATION_DIR / "validation_report.json",
        VALIDATION_DIR / "kappa_report.json",
        VALIDATION_DIR / "validation_report.md",
    ]
    available = [path for path in artifact_candidates if path.exists()]

    if not available:
        st.info("No downloadable artifacts found yet.")
        return

    for artifact in available:
        mime = "text/csv" if artifact.suffix == ".csv" else "application/octet-stream"
        st.download_button(
            label=f"Download {artifact.name}",
            data=artifact.read_bytes(),
            file_name=artifact.name,
            mime=mime,
        )


def main() -> None:
    st.set_page_config(page_title="BiasEval Dashboard", layout="wide")
    st.title("BiasEval Dashboard")

    data = _load_data()
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Module deep dives",
            "Prompt-level explorer",
            "Statistical validation",
            "Downloads",
        ],
    )

    if page == "Overview":
        _render_overview(data)
    elif page == "Module deep dives":
        _render_stereotype_deep_dive(data["stereotype"])
        st.divider()
        _render_representation_deep_dive(data["representation"])
        st.divider()
        _render_counterfactual_deep_dive(data["counterfactual"])
    elif page == "Prompt-level explorer":
        _render_prompt_explorer(data)
    elif page == "Statistical validation":
        _render_validation_section()
    elif page == "Downloads":
        _render_downloads()


if __name__ == "__main__":
    main()
