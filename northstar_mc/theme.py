"""Look of the NEFC 2026 deck: Cambria headings, Arial body, charcoal and gray, teal accent."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Deck theme "NEFC Restrained" plus the project colors used on its charts.
INK = "#1A1A1A"
INK_2 = "#4A4A4A"
MUTED = "#8C8C8C"
RULE = "#D9D9D9"
PANEL = "#F2F2F2"
TEAL = "#00736B"
TEAL_TINT = "#EAF4F3"
RED = "#B23A3A"
PROJECT_COLORS = {"Bison": TEAL, "Fennec": "#2F6DB5", "Puma": RED}
PROJECT_SYMBOLS = {"Bison": "circle", "Fennec": "square", "Puma": "diamond"}
LOW_COLOR, HIGH_COLOR = INK_2, "#B5B5B5"  # tornado: input at P10 / P90

HEADING_FONT = "Cambria, 'Source Serif 4', Georgia, 'Times New Roman', serif"
BODY_FONT = "Arial, Helvetica, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&display=swap');
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{ font-family: {BODY_FONT}; }}
h1, h2, h3, h4, h1 *, h2 *, h3 *, [data-testid="stMetricValue"] {{ font-family: {HEADING_FONT} !important; color: {INK}; font-weight: 700; }}
h1 {{ font-size: 2.1rem; line-height: 1.2; }}
h3 {{ font-size: 1.35rem; }}
.kicker {{ font-family: {BODY_FONT}; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
          color: {MUTED}; font-weight: 700; margin: 1.2rem 0 0.1rem 0; }}
.legend-dots {{ font-family: {BODY_FONT}; font-size: 0.85rem; font-weight: 700; }}
.legend-dots span {{ margin-right: 1.2rem; }}
.badge {{ border-top: 1px solid {RULE}; padding-top: 0.6rem; }}
.badge .name {{ font-size: 0.9rem; font-weight: 700; }}
.badge .status {{ font-size: 0.72rem; letter-spacing: 0.12em; font-weight: 700; margin-left: 0.4rem; }}
.badge .big {{ font-family: {HEADING_FONT} !important; font-size: 2.3rem; font-weight: 700; line-height: 1.15; }}
.badge .sub {{ color: {INK_2}; font-size: 0.85rem; }}
.badge .note {{ color: {MUTED}; font-size: 0.78rem; font-style: italic; margin-top: 0.25rem; }}
.source {{ color: {MUTED}; font-size: 0.7rem; border-top: 1px solid {RULE}; padding-top: 0.4rem; margin-top: 1rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; }}
</style>
"""


def kicker(text: str) -> str:
    return f'<div class="kicker">{text}</div>'


def legend_dots() -> str:
    dots = "".join(f'<span style="color:{c}">● {p}</span>' for p, c in PROJECT_COLORS.items())
    return f'<div class="legend-dots">{dots}</div>'


def register_template() -> None:
    """Plotly template in the deck's style: white, no plot frame, gray axes, Cambria titles."""
    axis = dict(showgrid=False, zeroline=False, showline=True, linecolor=MUTED, ticks="outside",
                tickcolor=MUTED, tickfont=dict(color=MUTED, size=12), title_font=dict(color=INK_2, size=12))
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=BODY_FONT, color=INK_2, size=12),
        title=dict(font=dict(family=HEADING_FONT, color=INK, size=17), x=0, xanchor="left"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        colorway=list(PROJECT_COLORS.values()),
        xaxis=axis, yaxis=axis | dict(showline=False, showgrid=True, gridcolor=PANEL),
        legend=dict(font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font=dict(family=BODY_FONT, color=INK)),
    )
    pio.templates["nefc"] = t
    pio.templates.default = "nefc"


def style_figure(fig: go.Figure) -> go.Figure:
    """Apply the deck look as explicit layout values, which win over any template Streamlit injects."""
    t = pio.templates["nefc"].layout
    fig.update_layout(template="nefc", font=t.font, paper_bgcolor=t.paper_bgcolor, plot_bgcolor=t.plot_bgcolor,
                      title_font=t.title.font, hoverlabel=t.hoverlabel)
    fig.update_xaxes(t.xaxis.to_plotly_json(), automargin=True)
    fig.update_yaxes(t.yaxis.to_plotly_json(), automargin=True)
    if fig.layout.margin.l is None or fig.layout.margin.l < 60:
        fig.update_layout(margin_l=60)
    return fig
