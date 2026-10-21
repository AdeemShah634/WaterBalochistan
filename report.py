"""One-click district PDF summary (FPDF2).

Deliberately plain-language: the audience is NGO field teams and local
government offices, not hydrologists. Everything on the page comes from
`analysis.district_summary`, so the report and the dashboard can never disagree.
"""
from __future__ import annotations

import io
from datetime import date

from . import analysis, config

try:
    from fpdf import FPDF

    PDF_AVAILABLE = True
    PDF_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    # fpdf2 only drives the PDF export. A missing optional dependency must
    # disable that one button, never take the whole dashboard down with it.
    FPDF = object
    PDF_AVAILABLE = False
    PDF_IMPORT_ERROR = str(exc)

_HEX = lambda h: tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))  # noqa: E731

PLAIN_LANGUAGE = {
    "Severe Deficit": (
        "Groundwater is among the lowest ever recorded for this district in this "
        "month. Treat as an urgent water-stress situation."
    ),
    "Critical": (
        "Groundwater is well below what is normal for this district in this month. "
        "Conditions warrant close monitoring and intervention planning."
    ),
    "Moderate Stress": (
        "Groundwater is somewhat below normal for this district in this month. "
        "Worth watching, but not yet critical."
    ),
    "Stable": (
        "Groundwater is within the normal range for this district in this month. "
        "No drought classification applies."
    ),
}


def _report_class():
    """Build the report class lazily, so importing this module never needs fpdf2."""
    if not PDF_AVAILABLE:
        raise RuntimeError(
            "PDF export needs fpdf2, which is not installed in the interpreter "
            f"running this app ({PDF_IMPORT_ERROR}). Install it with: pip install fpdf2"
        )

    class _Report(FPDF):
        def __init__(self, district: str, month: str):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.district = district
            self.month = month
            self.set_auto_page_break(auto=True, margin=18)

        def header(self) -> None:
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(20, 40, 70)
            self.cell(0, 8, "Groundwater District Summary", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(90, 90, 90)
            self.cell(
                0,
                5,
                "Groundwater Sustainability Monitoring & Forecasting System for Balochistan",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.set_draw_color(200, 205, 215)
            self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
            self.ln(6)
            self.set_text_color(0, 0, 0)

        def footer(self) -> None:
            self.set_y(-14)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(120, 120, 120)
            self.cell(
                0,
                4,
                "Values derive from a 0.25 deg GRACE-assimilated land surface model (NASA GLDAS-2.2 "
                "CLSM) and CHIRPS rainfall.",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.cell(
                0,
                4,
                "They are district-scale averages, not individual well or aquifer measurements - "
                f"a screening tool, not a substitute for ground observation.   Page {self.page_no()}",
            )

        # -- building blocks ---------------------------------------------------
        def section(self, title: str) -> None:
            self.ln(3)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(20, 40, 70)
            self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.set_font("Helvetica", "", 10)

        def kv_row(self, label: str, value: str) -> None:
            self.set_font("Helvetica", "", 10)
            self.cell(62, 6, f"  {label}", border=0)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

        def banner(self, band: str, percentile: float) -> None:
            r, g, b = _HEX(config.RISK_COLORS[band])
            self.set_fill_color(r, g, b)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 11, f"  RISK LEVEL: {band.upper()}", fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_fill_color(245, 246, 248)
            self.set_text_color(40, 40, 40)
            self.set_font("Helvetica", "", 9.5)
            self.cell(
                0,
                7,
                f"  {percentile:.0f}th percentile of this district's own record for the same "
                f"calendar month  |  USDM equivalent: {config.RISK_USDM[band]}",
                fill=True,
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.set_text_color(0, 0, 0)

    return _Report


def build_district_report(district: str, month: str | None = None) -> bytes:
    """Render the one-page district summary and return the PDF bytes."""
    s = analysis.district_summary(district, month)
    pdf = _report_class()(district, s["month"])
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, s["district"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(90, 90, 90)
    period = "observed data" if s["period"] == "observed" else "MODEL FORECAST"
    pdf.cell(
        0,
        5,
        f"Reference month: {s['month']} ({period})    Report generated: {date.today():%d %b %Y}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.banner(s["risk_band"], s["percentile"])
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, PLAIN_LANGUAGE[s["risk_band"]])

    pdf.section("Groundwater status")
    pdf.kv_row("Storage this month", f"{s['gws_mm']:.1f} mm")
    pdf.kv_row("Percentile vs own history", f"{s['percentile']:.0f}th")
    pdf.kv_row(
        "Long-term trend",
        f"{s['trend_mm_per_year']:+.2f} mm per year "
        f"({'declining' if s['trend_mm_per_year'] < 0 else 'rising'})",
    )
    pdf.kv_row(
        "Historical range",
        f"{s['min_gws_mm']:.1f} - {s['max_gws_mm']:.1f} mm (mean {s['mean_gws_mm']:.1f} mm)",
    )
    pdf.kv_row(
        "Months at Severe Deficit",
        f"{s['months_severe_observed']} of {s['months_observed']} observed months",
    )

    pdf.section("Rainfall (secondary indicator)")
    if s["rain_12m_mm"] is None:
        pdf.multi_cell(
            0,
            5.5,
            "  Rainfall is measured, not forecast, so the 12-month rainfall indicator is not "
            "available for a forecast month.",
        )
    else:
        pdf.kv_row("Rainfall, last 12 months", f"{s['rain_12m_mm']:.1f} mm")
        pdf.kv_row("Percentile vs own history", f"{s['rain_percentile']:.0f}th")
        pdf.kv_row(
            "Rainfall deficit flag",
            "YES - below the 20th percentile" if s["rain_deficit"] else "No",
        )

    pdf.section("Outlook")
    if s["forecast_end"] is not None:
        pdf.kv_row("Forecast horizon", f"through {s['forecast_end']}")
        pdf.kv_row("Projected storage", f"{s['forecast_end_gws_mm']:.1f} mm")
        pdf.kv_row("Projected risk level", str(s["forecast_end_band"]))
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.multi_cell(
            0,
            4.5,
            "  Forecasts come from a Prophet time-series model fitted to this district's own "
            f"history and validated over {config.BACKTEST_ORIGINS} rolling "
            f"{config.HOLDOUT_MONTHS}-month test windows against seasonal-naive and ARIMA "
            f"baselines. Only the first {config.RELIABLE_FORECAST_MONTHS} months are "
            "presented as reliable.",
        )

    pdf.section("How to read the risk levels")
    pdf.set_font("Helvetica", "", 8.5)
    for label, upper, colour, usdm in config.RISK_BANDS:
        lower = 0.0
        for other_label, other_upper, _c, _u in config.RISK_BANDS:
            if other_upper < upper:
                lower = max(lower, other_upper)
        r, g, b = _HEX(colour)
        pdf.set_fill_color(r, g, b)
        pdf.cell(5, 5, "", fill=True)
        pdf.cell(
            0,
            5,
            f"  {label}: {lower:.0f}-{upper:.0f}th percentile   ({usdm})",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    out = pdf.output()
    return bytes(out)


def save_district_report(district: str, month: str | None = None, path=None):
    """Write the district report to disk and return the path."""
    path = path or (
        config.OUTPUT_DIR / f"{district.replace(' ', '_')}_{month or 'latest'}.pdf"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_district_report(district, month))
    return path


def report_stream(district: str, month: str | None = None) -> io.BytesIO:
    """PDF as a stream, for a Streamlit download button."""
    return io.BytesIO(build_district_report(district, month))
