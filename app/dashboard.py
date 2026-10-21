"""Tier 1 - Presentation Layer.

An interactive web GIS dashboard over the core analysis module. This file
contains no analysis logic of its own: every number on screen comes from
`src.analysis`, so the same figures would come back out of the REST API.

Run with:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src import analysis, config, db, pipeline, report

st.set_page_config(
    page_title="Balochistan Groundwater Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .band-chip {display:inline-block; padding:2px 10px; border-radius:10px;
              color:#fff; font-weight:600; font-size:0.85rem;}
  .period-observed {color:#2c6e49; font-weight:600;}
  .period-forecast {color:#9a5b00; font-weight:600;}
  .caption-sm {font-size:0.8rem; color:#666;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Data access (cached wrappers around the analysis layer)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def boot() -> bool:
    """Make sure the database exists before anything tries to read it."""
    if not db.is_built():
        return False
    analysis.refresh()
    return True


@st.cache_data(show_spinner=False)
def districts_geo() -> gpd.GeoDataFrame:
    return pipeline.load_districts()


@st.cache_data(show_spinner=False)
def timeline() -> list[str]:
    return analysis.timeline_months()


@st.cache_data(show_spinner=False)
def snapshot(month: str) -> pd.DataFrame:
    return analysis.risk_snapshot(month)


@st.cache_data(show_spinner=False)
def summary(month: str) -> dict:
    return analysis.province_summary(month)


@st.cache_data(show_spinner=False)
def series(district: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        analysis.groundwater_series(district),
        analysis.forecast_series(district),
        analysis.rainfall_series(district),
    )


@st.cache_data(show_spinner=False)
def risk_history(district: str) -> pd.DataFrame:
    return analysis.district_risk_history(district)


@st.cache_data(show_spinner=False)
def metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    return analysis.metrics_summary(), analysis.model_metrics()


@st.cache_data(show_spinner=False)
def origin_breakdown() -> pd.DataFrame:
    return analysis.backtest_by_origin()


@st.cache_data(show_spinner=False)
def config_comparison() -> pd.DataFrame:
    return analysis.config_comparison()


@st.cache_data(show_spinner=False)
def quality() -> pd.DataFrame:
    return analysis.data_quality_report()


# ==========================================================================
# Helpers
# ==========================================================================
def to_dates(months: pd.Series) -> pd.Series:
    return pd.PeriodIndex(months, freq="M").to_timestamp()


def band_chip(band: str) -> str:
    return (
        f'<span class="band-chip" style="background:{config.RISK_COLORS[band]}">'
        f"{band}</span>"
    )


def build_map(month: str, gdf: gpd.GeoDataFrame, snap: pd.DataFrame, highlight: str | None):
    """District risk choropleth: colour is the risk band, not the raw value."""
    merged = gdf.merge(snap, on="district", how="left")
    merged["risk_band"] = merged["risk_band"].fillna("Stable")

    fmap = folium.Map(location=[28.5, 65.8], zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Basemap",
        control=False,
    ).add_to(fmap)

    def style(feature):
        band = feature["properties"]["risk_band"]
        is_sel = highlight is not None and feature["properties"]["district"] == highlight
        return {
            "fillColor": config.RISK_COLORS.get(band, "#cccccc"),
            "color": "#1a1a1a" if is_sel else "#ffffff",
            "weight": 3 if is_sel else 0.8,
            "fillOpacity": 0.82,
        }

    def highlight_fn(_feature):
        return {"weight": 3, "color": "#111111", "fillOpacity": 0.95}

    fields = ["district", "risk_band", "percentile", "gws_mm", "rain_deficit"]
    aliases = [
        "District:",
        "Risk band:",
        "Percentile (own history):",
        "Groundwater storage (mm):",
        "12-month rainfall deficit:",
    ]
    display = merged.copy()
    display["percentile"] = display["percentile"].round(0)
    display["gws_mm"] = display["gws_mm"].round(1)
    display["rain_deficit"] = display["rain_deficit"].map({1: "Yes", 0: "No"}).fillna("n/a")

    folium.GeoJson(
        display,
        name=f"Risk {month}",
        style_function=style,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
        popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
    ).add_to(fmap)

    legend_rows = "".join(
        f'<div style="margin:2px 0"><span style="background:{colour};width:14px;'
        f'height:14px;display:inline-block;border:1px solid #999;vertical-align:middle">'
        f"</span> <span style='vertical-align:middle'>{label}</span></div>"
        for label, _u, colour, _usdm in config.RISK_BANDS
    )
    legend = f"""
    <div style="position:fixed; bottom:22px; left:12px; z-index:9999;
                background:rgba(255,255,255,0.94); padding:8px 12px; border:1px solid #bbb;
                border-radius:6px; font-size:12px; font-family:sans-serif;">
      <div style="font-weight:700; margin-bottom:4px">Groundwater Risk &mdash; {month}</div>
      {legend_rows}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))

    minx, miny, maxx, maxy = merged.total_bounds
    fmap.fit_bounds([[miny, minx], [maxy, maxx]], padding=(4, 4))
    return fmap


def groundwater_chart(district: str, gw: pd.DataFrame, fc: pd.DataFrame, marker: str) -> go.Figure:
    fig = go.Figure()
    if not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=list(to_dates(fc["month"])) + list(to_dates(fc["month"]))[::-1],
                y=list(fc["yhat_upper"]) + list(fc["yhat_lower"])[::-1],
                fill="toself",
                fillcolor="rgba(214,140,20,0.18)",
                line=dict(width=0),
                hoverinfo="skip",
                name="80% uncertainty interval",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=to_dates(gw["month"]),
            y=gw["gws_mm"],
            mode="lines",
            name="Observed (GLDAS-2.2 GRACE-DA)",
            line=dict(color="#1f6fb2", width=1.6),
        )
    )
    if not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=to_dates(fc["month"]),
                y=fc["yhat"],
                mode="lines",
                name="Forecast (Prophet)",
                line=dict(color="#d68c14", width=2, dash="dash"),
            )
        )
        split = to_dates(pd.Series([fc["month"].iloc[0]]))[0]
        fig.add_vline(x=split, line=dict(color="#888", width=1, dash="dot"))
        fig.add_annotation(
            x=split, y=1.02, yref="paper", text="forecast begins", showarrow=False,
            font=dict(size=10, color="#777"), xanchor="left",
        )
    fig.add_vline(
        x=to_dates(pd.Series([marker]))[0], line=dict(color="#c1121f", width=1.4)
    )
    fig.update_layout(
        title=dict(
            text=f"{district} - groundwater storage, observed and forecast",
            yref="container", y=0.97, yanchor="top", x=0, xanchor="left",
            font=dict(size=15),
        ),
        yaxis_title="Groundwater storage (mm)",
        xaxis_title=None,
        height=450,
        margin=dict(l=10, r=10, t=92, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def rainfall_chart(district: str, rain: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=to_dates(rain["month"]),
            y=rain["rain_mm"],
            name="Monthly rainfall (CHIRPS)",
            marker_color="#89c2d9",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=to_dates(rain["month"]),
            y=rain["rain_12m_mm"],
            name="12-month rolling total",
            yaxis="y2",
            line=dict(color="#014f86", width=2),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{district} - rainfall (secondary indicator)",
            yref="container", y=0.97, yanchor="top", x=0, xanchor="left",
            font=dict(size=15),
        ),
        yaxis=dict(title="Monthly rainfall (mm)"),
        yaxis2=dict(title="12-month total (mm)", overlaying="y", side="right", showgrid=False),
        height=360,
        margin=dict(l=10, r=10, t=92, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.1,
    )
    return fig


def percentile_chart(district: str, hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, upper, colour, _usdm in config.RISK_BANDS:
        lower = max([u for _l, u, _c, _s in config.RISK_BANDS if u < upper], default=0.0)
        fig.add_hrect(
            y0=lower, y1=upper, fillcolor=colour, opacity=0.16, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=9,
        )
    for period, dash in (("observed", "solid"), ("forecast", "dash")):
        part = hist[hist["period"] == period]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=to_dates(part["month"]),
                y=part["percentile"],
                mode="lines",
                name=period.capitalize(),
                line=dict(color="#22223b", width=1.4, dash=dash),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"{district} - risk percentile vs its own same-month history",
            yref="container", y=0.97, yanchor="top", x=0, xanchor="left",
            font=dict(size=15),
        ),
        yaxis=dict(title="Percentile", range=[0, 100]),
        height=360,
        margin=dict(l=10, r=10, t=92, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ==========================================================================
# App
# ==========================================================================
if not boot():
    st.title("💧 Balochistan Groundwater Monitor")
    st.error("The database has not been built yet.")
    st.markdown(
        "Run the pipeline once to extract, validate, forecast and score the data:\n\n"
        "```\npython build_data.py\n```\n\n"
        "then reload this page."
    )
    st.stop()

months = timeline()
observed_set = set(analysis.observed_months())

# ---------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.markdown("### 💧 Groundwater Monitor")
    st.caption("Balochistan · district-level monitoring & forecasting")

    default_idx = months.index(analysis.latest_observed_month())
    month = st.select_slider(
        "Timeline",
        options=months,
        value=months[default_idx],
        help="The slider spans the observed record and the 3-year forecast horizon.",
    )
    is_observed = month in observed_set
    st.markdown(
        f'<span class="period-{"observed" if is_observed else "forecast"}">'
        f'{"● Observed data" if is_observed else "◑ Model forecast"}</span>'
        f' &nbsp;<span class="caption-sm">({month})</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    district = st.selectbox("District", analysis.list_districts(), index=0)

    st.divider()
    st.markdown("**Risk bands** (percentile of the district's own same-month history)")
    for label, upper, colour, usdm in config.RISK_BANDS:
        lower = max([u for _l, u, _c, _s in config.RISK_BANDS if u < upper], default=0.0)
        st.markdown(
            f'<div style="margin:3px 0"><span style="background:{colour};width:13px;height:13px;'
            f'display:inline-block;border-radius:3px;vertical-align:middle"></span> '
            f'<b>{label}</b> <span class="caption-sm">{lower:.0f}–{upper:.0f}th · {usdm}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        f"Record {analysis.observed_months()[0]} → {analysis.latest_observed_month()} · "
        f"{len(analysis.list_districts())} districts · forecast to {months[-1]}"
    )

# ---------------------------- Header --------------------------------------
st.title("Groundwater Sustainability Monitoring & Forecasting System")
st.caption(
    "Balochistan · GRACE-assimilated groundwater storage (NASA GLDAS-2.2 CLSM) and "
    "CHIRPS rainfall, aggregated per district"
)

tab_map, tab_district, tab_models, tab_data = st.tabs(
    ["🗺️ Risk Map", "📈 District Detail", "🎯 Model Validation", "🧾 Data & Method"]
)

# ---------------------------- Tab 1: Map ----------------------------------
with tab_map:
    snap = snapshot(month)
    prov = summary(month)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Month", month, "observed" if is_observed else "forecast", delta_color="off"
    )
    c2.metric("Severe Deficit", prov["band_counts"]["Severe Deficit"])
    c3.metric("Critical", prov["band_counts"]["Critical"])
    c4.metric("Moderate Stress", prov["band_counts"]["Moderate Stress"])
    c5.metric("Stable", prov["band_counts"]["Stable"])

    if not is_observed:
        st.warning(
            f"**{month} is a model forecast**, not an observation. Only the first "
            f"{config.RELIABLE_FORECAST_MONTHS} forecast months are presented as reliable.",
            icon="⚠️",
        )

    left, right = st.columns([1.85, 1.15])
    with left:
        fmap = build_map(month, districts_geo(), snap, district)
        st_folium(fmap, width=None, height=560, returned_objects=[], key=f"map-{month}-{district}")
    with right:
        deficit_count = int(snap["rain_deficit"].sum())
        st.markdown(f"**District ranking — {month}**")
        # The raw mm value is in the map tooltip and on the District Detail tab;
        # the ranking table stays on percentiles so all five columns fit.
        table = snap[
            ["district", "risk_band", "percentile", "rain_percentile", "rain_deficit"]
        ].copy()
        table["percentile"] = table["percentile"].round(0)
        # The rainfall columns are undefined in two cases: over the forecast
        # horizon (rainfall is measured, not forecast) and in the first 11 months
        # of the record (the 12-month window is not yet complete). Say "n/a"
        # rather than leaving a blank cell or printing a raw None.
        table["rain_percentile"] = [
            "n/a" if pd.isna(v) else f"{v:.0f}" for v in snap["rain_percentile"]
        ]
        table["rain_deficit"] = [
            "n/a" if pd.isna(pct) else ("⚠️ Yes" if flag else "No")
            for pct, flag in zip(snap["rain_percentile"], snap["rain_deficit"])
        ]
        rain_scored = int(snap["rain_percentile"].notna().sum())
        st.dataframe(
            table.rename(
                columns={
                    "district": "District",
                    "risk_band": "Risk band",
                    "percentile": "GWS %ile",
                    "rain_percentile": "Rain %ile",
                    "rain_deficit": "Rain deficit",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=505,
        )
        if not is_observed:
            st.caption(
                "Rainfall is measured, not forecast, so the rainfall columns are only "
                "defined over the observed period."
            )
        elif rain_scored == 0:
            st.caption(
                f"The rainfall indicator needs a full {config.RAINFALL_WINDOW_MONTHS}-month "
                "window, so it is not defined this early in the record."
            )
        else:
            st.caption(
                f"**Rain deficit** flags a trailing {config.RAINFALL_WINDOW_MONTHS}-month "
                f"rainfall total below the {config.RAINFALL_DEFICIT_PERCENTILE:.0f}th percentile "
                f"of the district's own same-month history — {deficit_count} of {rain_scored} "
                "districts this month."
            )

    st.markdown("#### Province-wide risk over time")
    tl = analysis.risk_timeline()
    fig = go.Figure()
    for label, _u, colour, _s in reversed(config.RISK_BANDS):
        fig.add_trace(
            go.Scatter(
                x=to_dates(tl["month"]),
                y=tl[label],
                stackgroup="one",
                name=label,
                mode="none",
                fillcolor=colour,
            )
        )
    first_forecast = analysis.forecast_months()[0]
    fig.add_vline(x=to_dates(pd.Series([first_forecast]))[0], line=dict(color="#333", dash="dot"))
    fig.add_vline(x=to_dates(pd.Series([month]))[0], line=dict(color="#c1121f", width=1.4))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Districts",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Dotted line marks where the observed record ends and the forecast begins; "
        "the red line is the month selected in the sidebar."
    )

# ---------------------------- Tab 2: District -----------------------------
with tab_district:
    gw, fc, rain = series(district)
    hist = risk_history(district)
    detail = analysis.district_summary(district, month)

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(f"### {district}")
        st.markdown(
            f"{band_chip(detail['risk_band'])} &nbsp; "
            f"<span class='caption-sm'>{month} · {detail['percentile']:.0f}th percentile · "
            f"USDM equivalent: {detail['usdm_equivalent']}</span>",
            unsafe_allow_html=True,
        )
    with head_r:
        if report.PDF_AVAILABLE:
            st.download_button(
                "📄 Download PDF summary",
                data=report.build_district_report(district, month),
                file_name=f"{district.replace(' ', '_')}_{month}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button(
                "📄 PDF export unavailable",
                disabled=True,
                use_container_width=True,
                help=(
                    "Install fpdf2 in the interpreter running this app "
                    "(pip install fpdf2), then restart it."
                ),
            )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Groundwater storage", f"{detail['gws_mm']:.1f} mm")
    m2.metric(
        "Trend",
        f"{detail['trend_mm_per_year']:+.2f} mm/yr",
        delta="declining" if detail["trend_mm_per_year"] < 0 else "rising",
        delta_color="inverse" if detail["trend_mm_per_year"] < 0 else "normal",
    )
    m3.metric(
        "12-month rainfall",
        "n/a (forecast)" if detail["rain_12m_mm"] is None else f"{detail['rain_12m_mm']:.0f} mm",
        delta=None if detail["rain_12m_mm"] is None
        else ("deficit" if detail["rain_deficit"] else "normal"),
        delta_color="inverse" if detail["rain_deficit"] else "off",
    )
    m4.metric(
        "Months at Severe Deficit",
        f"{detail['months_severe_observed']} / {detail['months_observed']}",
    )

    st.plotly_chart(
        groundwater_chart(district, gw, fc, month), use_container_width=True
    )
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(percentile_chart(district, hist), use_container_width=True)
    with cr:
        st.plotly_chart(rainfall_chart(district, rain), use_container_width=True)

    with st.expander("Forecast accuracy for this district (rolling-origin backtest)"):
        _, per_district = metrics()
        dm = per_district[per_district["district"] == district][["model", "rmse", "mae"]]
        st.dataframe(
            dm.rename(columns={"model": "Model", "rmse": "RMSE (mm)", "mae": "MAE (mm)"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Averaged over {config.BACKTEST_ORIGINS} rolling "
            f"{config.HOLDOUT_MONTHS}-month test windows, each forecast from a training "
            "set ending before it. Lower is better."
        )

# ---------------------------- Tab 3: Models -------------------------------
with tab_models:
    st.markdown("### Forecast validation")
    st.markdown(
        f"Accuracy is measured by **rolling-origin backtest**: "
        f"{config.BACKTEST_ORIGINS} separate {config.HOLDOUT_MONTHS}-month test "
        f"windows, {config.BACKTEST_STRIDE} months apart, each forecast from a "
        "training set that ends before it. Prophet is scored against a "
        "**seasonal-naive** baseline (repeat the value 12 months earlier) and a "
        f"**SARIMA{config.ARIMA_ORDER}{config.ARIMA_SEASONAL_ORDER}** baseline."
    )
    summary_df, per_district = metrics()

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(
            summary_df.rename(
                columns={
                    "model": "Model",
                    "rmse": "Mean RMSE (mm)",
                    "mae": "Mean MAE (mm)",
                    "districts_best_rmse": "Districts where best",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )
        ranked = summary_df.sort_values("rmse")
        best, second = ranked.iloc[0], ranked.iloc[1]
        margin = 100 * (second["rmse"] - best["rmse"]) / second["rmse"]
        st.info(
            f"**{best['model']}** has the lowest mean RMSE ({best['rmse']:.2f} mm), "
            f"{margin:.0f}% ahead of **{second['model']}** ({second['rmse']:.2f} mm), "
            f"averaged over {config.BACKTEST_ORIGINS} test windows and 26 districts.",
            icon="🔎",
        )
    with c2:
        pivot = per_district.pivot(index="district", columns="model", values="rmse")
        fig = go.Figure()
        colours = {"prophet": "#d68c14", "arima": "#5e548e", "seasonal_naive": "#1f6fb2"}
        for model in ["prophet", "arima", "seasonal_naive"]:
            if model in pivot.columns:
                fig.add_trace(
                    go.Bar(
                        x=pivot.index, y=pivot[model], name=model,
                        marker_color=colours.get(model),
                    )
                )
        fig.update_layout(
            barmode="group",
            height=420,
            yaxis_title="Mean RMSE (mm) across test windows",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Why more than one test window")
    st.markdown(
        "A single holdout is not enough evidence. Scored only on the most recent "
        "24 months, the seasonal-naive baseline looks competitive — that window "
        "happens to suit it. The ranking changes from window to window — the "
        "seasonal-naive baseline even wins the window starting 2021-06 — which is "
        "why the figures above average over all of them."
    )
    by_origin = origin_breakdown()
    oc1, oc2 = st.columns([1.3, 1])
    with oc1:
        fig = go.Figure()
        for model in ["prophet", "arima", "seasonal_naive"]:
            if model in by_origin.columns:
                fig.add_trace(
                    go.Bar(
                        x=by_origin["origin_month"],
                        y=by_origin[model],
                        name=model,
                        marker_color=colours.get(model),
                    )
                )
        fig.update_layout(
            barmode="group",
            title=dict(
                text="Mean RMSE by test window (window start month)",
                yref="container", y=0.97, yanchor="top", x=0, xanchor="left",
                font=dict(size=14),
            ),
            yaxis_title="Mean RMSE (mm)",
            height=340,
            margin=dict(l=10, r=10, t=80, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    with oc2:
        st.dataframe(
            by_origin.rename(columns={"origin_month": "Test window starts"}).round(2),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    st.markdown("#### Why this Prophet configuration")
    st.markdown(
        "Prophet's stock `changepoint_prior_scale=0.05` lets the trend bend freely "
        "at 25 changepoints. Extrapolated one to three years that produces a "
        "spurious slope, so the trend is deliberately constrained to "
        f"`{config.PROPHET_PARAMS['changepoint_prior_scale']}`. Each alternative was "
        "scored over the same test windows:"
    )
    configs_df = config_comparison()
    cc1, cc2 = st.columns([1, 1.2])
    with cc1:
        st.dataframe(
            configs_df.rename(
                columns={"config": "Configuration", "rmse": "Mean RMSE (mm)", "mae": "Mean MAE (mm)"}
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )
    with cc2:
        fig = go.Figure(
            go.Bar(
                x=configs_df["rmse"],
                y=configs_df["config"],
                orientation="h",
                marker_color=[
                    "#2c6e49" if "chosen" in c else "#9aa5b1" for c in configs_df["config"]
                ],
                text=configs_df["rmse"].round(2),
                textposition="outside",
            )
        )
        fig.update_layout(
            height=260,
            xaxis_title="Mean RMSE (mm)",
            margin=dict(l=10, r=40, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Per-district metrics")
    st.dataframe(
        per_district.pivot(index="district", columns="model", values=["rmse", "mae"]).round(2),
        use_container_width=True,
    )

# ---------------------------- Tab 4: Data ---------------------------------
with tab_data:
    st.markdown("### Data quality report")
    q = quality()
    icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    q_display = q.copy()
    q_display["status"] = q_display["status"].map(lambda s: f"{icons.get(s, '')} {s}")
    st.dataframe(
        q_display.rename(
            columns={"check_name": "Check", "status": "Status", "detail": "Detail"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Method")
        st.markdown(
            f"""
**Extract** — Monthly mean groundwater storage (`GWS_tavg`, NASA GLDAS-2.2 CLSM
GRACE-DA, 0.25°) and monthly total rainfall (CHIRPS, 0.05°) are aggregated per
district by zonal statistics and exported as CSV; district boundaries are
exported as GeoJSON.

**Process** — The exports are validated in Pandas (name matching against the
boundary file, coverage and gap checks, null and range checks) and loaded into
SQLite alongside the district geometry, which is read with GeoPandas.

**Forecast** — A Prophet model per district projects
{config.FORECAST_MONTHS} months ahead with 80% uncertainty intervals, of which
the first {config.RELIABLE_FORECAST_MONTHS} are presented as reliable. The trend
is deliberately constrained
(`changepoint_prior_scale={config.PROPHET_PARAMS['changepoint_prior_scale']}`)
because a freely bending trend extrapolates a spurious slope over a multi-year
horizon. Accuracy is measured by rolling-origin backtest over
{config.BACKTEST_ORIGINS} separate {config.HOLDOUT_MONTHS}-month test windows
against seasonal-naive and ARIMA baselines using RMSE and MAE.

**Risk index** — Each value is ranked against that district's *own* record for
the *same calendar month*, so ordinary seasonal variation is not mistaken for
depletion. The percentile is mapped onto four bands with cut-offs following the
U.S. Drought Monitor convention. A trailing
{config.RAINFALL_WINDOW_MONTHS}-month rainfall total below the
{config.RAINFALL_DEFICIT_PERCENTILE:.0f}th percentile is carried as a secondary
indicator.
            """
        )
    with c2:
        st.markdown("### Scope and limitations")
        st.warning(
            "Groundwater values derive from a 0.25° GRACE-assimilated land surface "
            "model and represent **district-scale averages, not individual well or "
            "aquifer measurements**. The system is a screening and prioritization "
            "tool, not a replacement for ground observation.",
            icon="⚠️",
        )
        st.markdown(
            """
- Administrative boundaries come from a public dataset whose vintage is a
  declared limitation; district names are matched exactly against it.
- Rainfall is measured, not forecast, so the rainfall deficit indicator is only
  defined over the observed period.
- Forecast bands are scored against the *observed* historical distribution, so a
  band means the same thing on both sides of the timeline.
            """
        )
        st.markdown("### District geometry (GeoPandas)")
        attrs = analysis.districts_table().round(3)
        st.dataframe(
            attrs.rename(
                columns={
                    "district": "District",
                    "area_km2": "Area (km²)",
                    "centroid_lat": "Centroid lat",
                    "centroid_lon": "Centroid lon",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=260,
        )

    st.markdown("### Download")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Risk index (all districts, all months) — CSV",
            data=db.read_table("risk_index").to_csv(index=False).encode(),
            file_name="balochistan_risk_index.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ Model metrics — CSV",
            data=metrics()[1].to_csv(index=False).encode(),
            file_name="balochistan_model_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
