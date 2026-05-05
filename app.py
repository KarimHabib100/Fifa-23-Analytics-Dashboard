import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Data Loading ──────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("data/cleaned_players.csv")
    LEAGUES = sorted(df["league_name"].dropna().unique().tolist())
    AGE_MIN = int(df["age"].min())
    AGE_MAX = int(df["age"].max())
except FileNotFoundError:
    df = pd.DataFrame()
    LEAGUES = []
    AGE_MIN, AGE_MAX = 16, 45

# ── Theme Constants ───────────────────────────────────────────────────────────
BG        = "#0f1117"
CARD_BG   = "#1a1d2e"
HEADER_BG = "#22253a"
ACCENT    = "#f0a500"
TEXT      = "#e0e0e0"
BORDER    = "#2c2f45"

PLOT_BG  = "#161929"
C_TEXT   = "#ffffff"
GRID_CLR = "rgba(255,255,255,0.08)"
AXIS_CLR = "rgba(255,255,255,0.35)"

# ── Color Palettes ────────────────────────────────────────────────────────────
POS_ORDER  = ["GK", "DEF", "MID", "FWD"]
POS_COLORS = {"GK": "#f0a500", "DEF": "#4fc3f7", "MID": "#81c784", "FWD": "#f48fb1"}

BLUE_SHADES  = ["#add8e6", "#7bb8d4", "#4d9abf", "#2a7fae", "#0d4a70"]
GREEN_SHADES = ["#90ee90", "#52c47c", "#228b22"]

# ── Common Layout Helper ──────────────────────────────────────────────────────
def _base(fig, title, xtitle, ytitle, margin=None, xgrid=False, yrangemode="tozero"):
    m = margin or dict(l=40, r=20, t=50, b=40)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=C_TEXT, family="Inter, Arial, sans-serif"),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=C_TEXT, size=13),
            x=0.5, xanchor="center", y=0.97,
        ),
        legend=dict(
            font=dict(color=C_TEXT, size=10),
            bgcolor="rgba(22,25,41,0.85)",
            bordercolor=AXIS_CLR,
            borderwidth=1,
            x=0.99, xanchor="right",
            y=0.99, yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor="#1a1d2e",
            font=dict(color="#ffffff", size=11, family="Inter, Arial, sans-serif"),
            bordercolor="#2c2f45",
        ),
        margin=m,
    )
    fig.update_xaxes(
        title_text=xtitle,
        title_font=dict(color=C_TEXT, size=11),
        tickfont=dict(color=C_TEXT),
        showgrid=xgrid, gridcolor=GRID_CLR,
        showline=True, linecolor=AXIS_CLR, mirror=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=ytitle,
        title_font=dict(color=C_TEXT, size=11),
        title_standoff=12,
        tickfont=dict(color=C_TEXT),
        showgrid=True, gridcolor=GRID_CLR,
        showline=True, linecolor=AXIS_CLR, mirror=True,
        rangemode=yrangemode,
        zeroline=False,
    )
    return fig


def empty_fig(msg="No data for selected filters"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=C_TEXT, size=15),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def build_col_chart(dff, metric="overall"):
    if dff.empty:
        return go.Figure()
    label = "Overall" if metric == "overall" else "Potential"

    top = (dff.groupby("club_name")[metric].mean()
              .sort_values(ascending=False).head(15).reset_index())
    top.columns = ["club_name", "avg_val"]
    top["avg_val"] = top["avg_val"].round(1)
    top["short"]   = top["club_name"].str[:10]

    max_val = top["avg_val"].max()
    colors  = ["lightgreen" if v == max_val else "lightblue" for v in top["avg_val"]]

    fig = go.Figure(go.Bar(
        x=top["short"], y=top["avg_val"],
        marker=dict(color=colors, line=dict(color="black", width=1.2)),
        text=[f"{v:.1f}" for v in top["avg_val"]],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=9),
        name=f"Avg {label}",
        customdata=top["club_name"],
        hovertemplate=f"%{{customdata}}<br>Avg {label}: %{{y:.1f}}<extra></extra>",
    ))
    fig.update_layout(yaxis_range=[0, max_val * 1.13], showlegend=True)
    _base(fig,
          f"Which Club Has the Highest Average {label} Rating?",
          "Club", f"Average {label} Rating",
          margin=dict(t=50, b=60, l=40, r=20))
    return fig


def build_bar_chart(dff, metric="overall"):
    if dff.empty:
        return go.Figure()
    label = "Overall" if metric == "overall" else "Potential"

    top = (dff.groupby("nationality_name")[metric].mean()
              .sort_values(ascending=False).head(15).reset_index())
    top.columns = ["nationality", "avg_val"]
    top["avg_val"] = top["avg_val"].round(1)
    top = top.sort_values("avg_val", ascending=True)

    max_val = top["avg_val"].max()
    colors  = ["lightgreen" if v == max_val else "lightblue" for v in top["avg_val"]]

    fig = go.Figure(go.Bar(
        x=top["avg_val"], y=top["nationality"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="black", width=1.2)),
        text=[f"{v:.1f}" for v in top["avg_val"]],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=9),
        name=f"Avg {label}",
        hovertemplate=f"%{{y}}<br>Avg {label}: %{{x:.1f}}<extra></extra>",
    ))
    fig.update_layout(xaxis_range=[0, max_val * 1.12])
    _base(fig,
          f"Top 15 Nationalities by Average {label} Rating",
          f"Average {label} Rating", "Nationality",
          xgrid=True, yrangemode="normal",
          margin=dict(t=50, b=40, l=130, r=80))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STACKED & CLUSTERED CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def build_stacked_col_chart(dff):
    if dff.empty:
        return go.Figure()

    sub = dff[dff["preferred_foot"].isin(["Left", "Right"])]
    if sub.empty:
        return go.Figure()

    positions_present = [p for p in POS_ORDER if p in sub["position_group"].values]
    grp = sub.groupby(["position_group", "preferred_foot"]).size().reset_index(name="count")

    totals  = grp.groupby("position_group")["count"].sum()
    max_pos = totals.idxmax() if not totals.empty else ""

    foot_palette = {
        "Left":  {"normal": "#add8e6", "highlight": "#90ee90"},
        "Right": {"normal": "#4682b4", "highlight": "#228b22"},
    }

    fig = go.Figure()
    for foot in ["Left", "Right"]:
        count_map = dict(zip(
            grp[grp["preferred_foot"] == foot]["position_group"],
            grp[grp["preferred_foot"] == foot]["count"],
        ))
        counts = [count_map.get(pos, 0) for pos in positions_present]
        bar_colors = [
            foot_palette[foot]["highlight"] if pos == max_pos else foot_palette[foot]["normal"]
            for pos in positions_present
        ]
        fig.add_trace(go.Bar(
            name=f"{foot} Foot",
            x=positions_present, y=counts,
            marker=dict(color=bar_colors, line=dict(color="black", width=0.8)),
            text=[f"{int(c):,}" if c > 0 else "" for c in counts],
            textposition="inside",
            textfont=dict(color="white", size=9),
        ))

    fig.update_layout(barmode="stack")
    _base(fig,
          "Player Composition by Position Group &amp; Preferred Foot",
          "Position Group", "Number of Players")
    return fig


def build_stacked_bar_chart(dff):
    if dff.empty:
        return go.Figure()

    top_leagues = dff["league_name"].value_counts().head(8).index.tolist()
    if not top_leagues:
        return go.Figure()

    sub = dff[dff["league_name"].isin(top_leagues)]
    grp = sub.groupby(["league_name", "position_group"]).size().reset_index(name="count")

    totals      = grp.groupby("league_name")["count"].sum().sort_values(ascending=True)
    max_league  = totals.idxmax() if not totals.empty else ""
    league_order = totals.index.tolist()

    pos_blues  = {"GK": "#add8e6", "DEF": "#7bb8d4", "MID": "#4d9abf", "FWD": "#2a7fae"}
    pos_greens = {"GK": "#90ee90", "DEF": "#52c47c", "MID": "#2e8b57", "FWD": "#155724"}

    fig = go.Figure()
    positions_present = [p for p in POS_ORDER if p in grp["position_group"].values]
    for pos in positions_present:
        count_map = dict(zip(
            grp[grp["position_group"] == pos]["league_name"],
            grp[grp["position_group"] == pos]["count"],
        ))
        counts = [count_map.get(lg, 0) for lg in league_order]
        bar_colors = [
            pos_greens[pos] if lg == max_league else pos_blues[pos]
            for lg in league_order
        ]
        fig.add_trace(go.Bar(
            name=pos, x=counts, y=league_order, orientation="h",
            marker=dict(color=bar_colors, line=dict(color="black", width=0.8)),
            text=[f"{int(c):,}" if c > 80 else "" for c in counts],
            textposition="inside",
            textfont=dict(color="white", size=8),
        ))

    fig.update_layout(barmode="stack")
    _base(fig,
          "Player Count Breakdown by Position Group per League",
          "Number of Players", "League",
          xgrid=True, yrangemode="normal",
          margin=dict(t=50, b=40, l=175, r=60))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True)
    return fig


def build_clustered_col_chart(dff):
    if dff.empty:
        return go.Figure()

    grp = (dff.groupby("position_group")[["overall", "potential"]]
              .mean().round(1).reset_index())
    positions_present = [p for p in POS_ORDER if p in grp["position_group"].values]
    grp = grp.set_index("position_group").reindex(positions_present).reset_index()

    max_pos = grp.loc[grp["overall"].idxmax(), "position_group"] if not grp.empty else ""

    fig = go.Figure()
    for metric, norm_col, hi_col in [
        ("overall",   "lightblue",  "lightgreen"),
        ("potential", "steelblue",  "seagreen"),
    ]:
        bar_colors = [
            hi_col if pos == max_pos else norm_col
            for pos in grp["position_group"]
        ]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=grp["position_group"], y=grp[metric],
            marker=dict(color=bar_colors, line=dict(color="black", width=1.2)),
            text=[f"{v:.1f}" for v in grp[metric]],
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
        ))

    y_max = grp["potential"].max() if not grp.empty else 100
    fig.update_layout(
        barmode="group",
        yaxis_range=[0, (y_max if not np.isnan(y_max) else 100) * 1.14],
    )
    _base(fig,
          "Overall vs Potential by Position Group — Who Has the Largest Gap?",
          "Position Group", "Average Rating")
    return fig


def build_clustered_bar_chart(dff):
    if dff.empty:
        return go.Figure()

    top_leagues = dff["league_name"].value_counts().head(5).index.tolist()
    if not top_leagues:
        return go.Figure()

    sub = dff[dff["league_name"].isin(top_leagues)]
    grp = (sub.groupby(["position_group", "league_name"])["wage_in_eur"]
              .mean().round(0).reset_index())

    league_avgs = grp.groupby("league_name")["wage_in_eur"].mean()
    max_league  = league_avgs.idxmax() if not league_avgs.empty else ""
    positions_present = [p for p in POS_ORDER if p in grp["position_group"].values]

    league_blues  = {lg: BLUE_SHADES[i]   for i, lg in enumerate(top_leagues)}
    league_greens = {lg: GREEN_SHADES[min(i, 2)] for i, lg in enumerate(top_leagues)}

    fig = go.Figure()
    for league in top_leagues:
        wage_map = dict(zip(
            grp[grp["league_name"] == league]["position_group"],
            grp[grp["league_name"] == league]["wage_in_eur"],
        ))
        wages  = [wage_map.get(pos, 0) / 1000 for pos in positions_present]
        color  = league_greens[league] if league == max_league else league_blues[league]
        fig.add_trace(go.Bar(
            name=league[:22],
            x=wages, y=positions_present, orientation="h",
            marker=dict(color=color, line=dict(color="black", width=0.8)),
            text=[f"€{w:.0f}K" if w > 0 else "" for w in wages],
            textposition="outside",
            textfont=dict(color=C_TEXT, size=8),
        ))

    x_max = (grp["wage_in_eur"].max() / 1000) if not grp.empty else 1
    fig.update_layout(barmode="group", xaxis_range=[0, x_max * 1.35])
    _base(fig,
          "Average Weekly Wage by Position — Top 5 Leagues Side-by-Side",
          "Average Weekly Wage (€K)", "Position Group",
          xgrid=True, yrangemode="normal",
          margin=dict(t=50, b=40, l=80, r=95))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RELATIONSHIP CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def build_scatter_chart(dff):
    if dff.empty:
        return go.Figure()

    sample = (dff.dropna(subset=["age", "overall", "position_group", "short_name"])
                 .sample(min(4000, len(dff)), random_state=42))
    if sample.empty:
        return go.Figure()

    try:
        fig = px.scatter(
            sample, x="age", y="overall",
            color="position_group",
            color_discrete_map=POS_COLORS,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="white",
            hover_name="short_name",
            hover_data={"age": True, "overall": True, "position_group": True},
            category_orders={"position_group": POS_ORDER},
            labels={"position_group": "Position"},
        )
    except Exception:
        fig = px.scatter(
            sample, x="age", y="overall",
            color="position_group",
            color_discrete_map=POS_COLORS,
            hover_name="short_name",
            category_orders={"position_group": POS_ORDER},
            labels={"position_group": "Position"},
        )

    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=4, line=dict(color="black", width=0.4), opacity=0.65),
    )
    fig.update_traces(
        selector=dict(mode="lines"),
        line=dict(width=2.5, dash="dash"),
    )
    fig.update_layout(yaxis_range=[0, 105])
    _base(fig, "Does Age Influence Overall Rating? — Correlation & Outliers",
          "Age", "Overall Rating")
    return fig


def build_bubble_chart(dff):
    if dff.empty:
        return go.Figure()

    sample = (dff.dropna(subset=["age", "market_value_in_eur", "potential",
                                  "position_group", "short_name"])
                 .sample(min(3000, len(dff)), random_state=42)
                 .copy())
    if sample.empty:
        return go.Figure()

    sample["value_M"] = sample["market_value_in_eur"] / 1_000_000

    fig = px.scatter(
        sample, x="age", y="value_M",
        size="potential", color="position_group",
        color_discrete_map=POS_COLORS,
        hover_name="short_name",
        hover_data={"club_name": True, "potential": True,
                    "value_M": ":.2f", "age": True},
        size_max=22,
        category_orders={"position_group": POS_ORDER},
        labels={"position_group": "Position", "value_M": "Market Value (€M)"},
    )
    fig.update_traces(marker=dict(line=dict(color="black", width=0.5), opacity=0.6))
    fig.update_layout(yaxis_tickformat=".1f")
    _base(fig,
          "Age vs Market Value — Bubble Size = Potential Rating",
          "Age", "Market Value (€M)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DISTRIBUTION CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def build_histogram_chart(dff):
    if dff.empty:
        return go.Figure()

    bins = np.arange(40, 101, 3)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    positions_present = [p for p in POS_ORDER if p in dff["position_group"].values]
    for pos in positions_present:
        vals = dff[dff["position_group"] == pos]["overall"].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        fig.add_trace(go.Bar(
            x=bin_centers, y=counts,
            name=pos,
            marker=dict(color=POS_COLORS[pos], line=dict(color="black", width=0.5)),
            opacity=0.6,
            text=[str(int(c)) if c > 0 else "" for c in counts],
            textposition="outside",
            textfont=dict(color=C_TEXT, size=7),
            hovertemplate=f"{pos} | Rating ~%{{x:.0f}} | Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(barmode="overlay")
    _base(fig,
          "Frequency Distribution of Overall Ratings — By Position Group",
          "Overall Rating", "Count")
    return fig


def build_box_chart(dff):
    if dff.empty:
        return go.Figure()

    sub = dff.dropna(subset=["wage_in_eur", "position_group"])
    positions_present = [p for p in POS_ORDER if p in sub["position_group"].values]

    fig = go.Figure()
    for pos in positions_present:
        vals = sub[sub["position_group"] == pos]["wage_in_eur"] / 1000
        fig.add_trace(go.Box(
            y=vals, name=pos,
            fillcolor="lightblue",
            line=dict(color="black", width=1.2),
            marker=dict(color="lightgreen", size=3,
                        line=dict(color="black", width=0.5)),
            boxpoints="outliers",
            whiskerwidth=0.6,
        ))

    for pos in positions_present:
        med = (sub[sub["position_group"] == pos]["wage_in_eur"] / 1000).median()
        if not np.isnan(med):
            fig.add_annotation(
                x=pos, y=med,
                text=f"€{med:.1f}K",
                showarrow=False,
                yanchor="bottom", yshift=7,
                font=dict(color=C_TEXT, size=8),
            )

    _base(fig,
          "Weekly Wage Distribution by Position Group — Five-Number Summary",
          "Position Group", "Weekly Wage (€K)")
    return fig


def build_violin_chart(dff):
    if dff.empty:
        return go.Figure()

    sub = dff[dff["preferred_foot"].isin(["Left", "Right"])].dropna(subset=["potential"])
    if sub.empty:
        return go.Figure()

    sampled = (sub.groupby("preferred_foot", group_keys=False)
                  .apply(lambda g: g.sample(min(800, len(g)), random_state=42)))

    foot_colors = {"Left": "#add8e6", "Right": "#4682b4"}
    feet_present = [f for f in ["Left", "Right"] if f in sampled["preferred_foot"].values]

    fig = go.Figure()
    for foot in feet_present:
        vals = sampled[sampled["preferred_foot"] == foot]["potential"]
        fig.add_trace(go.Violin(
            y=vals, name=f"{foot} Foot",
            fillcolor=foot_colors[foot],
            line_color="black",
            box_visible=True,
            box_fillcolor="rgba(255,255,255,0.2)",
            meanline_visible=False,
            points="all", jitter=0.35, pointpos=-1.8,
            marker=dict(size=1.8, color=foot_colors[foot], opacity=0.22),
            opacity=0.88,
        ))

    for foot in feet_present:
        med = sub[sub["preferred_foot"] == foot]["potential"].median()
        if not np.isnan(med):
            fig.add_annotation(
                x=f"{foot} Foot", y=med,
                text=f"Median: {int(med)}",
                showarrow=True, arrowhead=2,
                arrowcolor=C_TEXT, arrowwidth=1,
                font=dict(color=C_TEXT, size=9),
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor=AXIS_CLR, borderwidth=1,
                yanchor="bottom", ay=-34,
            )

    _base(fig,
          "Potential Distribution by Preferred Foot — KDE Silhouette",
          "Preferred Foot", "Potential Rating")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TIME-SERIES CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def build_line_chart(dff, metric="overall"):
    if dff.empty:
        return go.Figure()
    label = "Overall" if metric == "overall" else "Potential"

    grp = (dff.groupby(["age", "position_group"])[metric]
              .mean().round(2).reset_index().sort_values("age"))
    positions_present = [p for p in POS_ORDER if p in grp["position_group"].values]

    fig = go.Figure()
    for pos in positions_present:
        sub = grp[grp["position_group"] == pos]
        fig.add_trace(go.Scatter(
            x=sub["age"], y=sub[metric],
            name=pos, mode="lines+markers",
            line=dict(color=POS_COLORS[pos], width=2),
            marker=dict(size=4, color=POS_COLORS[pos],
                        line=dict(color="black", width=0.5)),
            hovertemplate=f"{pos} | Age: %{{x}} | Avg {label}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(yaxis_range=[0, 100])
    _base(fig,
          f"Average {label} Rating by Age — Trend per Position Group",
          "Age", f"Average {label} Rating")
    return fig


def build_area_chart(dff):
    if dff.empty:
        return go.Figure()

    age_cts = (dff.groupby("age").size().reset_index(name="count").sort_values("age"))
    age_cts["cumulative"] = age_cts["count"].cumsum()

    fig = go.Figure(go.Scatter(
        x=age_cts["age"], y=age_cts["cumulative"],
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.22)",
        line=dict(color="#4fc3f7", width=2.5),
        name="Cumulative Players", mode="lines",
        hovertemplate="Age: %{x}<br>Cumulative Players: %{y:,}<extra></extra>",
    ))
    _base(fig,
          "Cumulative Player Count by Age — Volume Builds Across Career Span",
          "Age", "Cumulative Number of Players")
    return fig


# ── Build Default Figures ─────────────────────────────────────────────────────
fig_col         = build_col_chart(df)
fig_bar         = build_bar_chart(df)
fig_stacked_col = build_stacked_col_chart(df)
fig_stacked_bar = build_stacked_bar_chart(df)
fig_clust_col   = build_clustered_col_chart(df)
fig_clust_bar   = build_clustered_bar_chart(df)
fig_scatter     = build_scatter_chart(df)
fig_bubble      = build_bubble_chart(df)
fig_histogram   = build_histogram_chart(df)
fig_box         = build_box_chart(df)
fig_violin      = build_violin_chart(df)
fig_line        = build_line_chart(df)
fig_area        = build_area_chart(df)

# ── Layout Helpers ────────────────────────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "marginBottom": "20px",
}
SECTION_STYLE = {
    "color": ACCENT, "fontWeight": "bold",
    "marginTop": "32px", "marginBottom": "14px",
    "borderBottom": f"2px solid {ACCENT}",
    "paddingBottom": "6px", "letterSpacing": "0.5px",
}
LABEL_STYLE = {"color": TEXT, "fontWeight": "bold", "marginBottom": "6px", "fontSize": "13px"}
GRAPH_CONFIG = {"displayModeBar": False}


def section_header(title):
    return html.H3(title, style=SECTION_STYLE)


def chart_card(title, graph_id, figure=None, height=420):
    return dbc.Card([
        dbc.CardHeader(
            html.H6(title, className="mb-0",
                    style={"color": TEXT, "fontWeight": "bold", "fontSize": "13px"}),
            style={"backgroundColor": HEADER_BG, "borderBottom": f"1px solid {BORDER}"}
        ),
        dbc.CardBody(
            dcc.Graph(
                id=graph_id,
                figure=figure if figure is not None else {},
                config=GRAPH_CONFIG,
                style={"height": f"{height}px"},
            ),
            style={"padding": "8px"}
        ),
    ], style=CARD_STYLE)


# ── App Init ──────────────────────────────────────────────────────────────────
_GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, _GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="FIFA 23 Analytics Dashboard",
)
server = app.server

# ── Dropdown / Slider Options ─────────────────────────────────────────────────
league_options   = [{"label": "All Leagues",   "value": "ALL"}] + [
    {"label": lg, "value": lg} for lg in LEAGUES
]
position_options = [{"label": "All Positions", "value": "ALL"}] + [
    {"label": pg, "value": pg} for pg in POS_ORDER
]

# ── Controls Panel ────────────────────────────────────────────────────────────
controls_panel = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("League", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="league-dropdown",
                    options=league_options, value="ALL", clearable=False,
                    style={"fontSize": "13px", "color": "black"},
                    optionHeight=35,
                ),
            ], xs=12, sm=6, md=3),

            dbc.Col([
                html.Label("Position Group", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="position-dropdown",
                    options=position_options, value="ALL", clearable=False,
                    style={"fontSize": "13px", "color": "black"},
                    optionHeight=35,
                ),
            ], xs=12, sm=6, md=3),

            dbc.Col([
                html.Label(f"Age Range  ({AGE_MIN} – {AGE_MAX})", style=LABEL_STYLE),
                dcc.RangeSlider(
                    id="age-slider",
                    min=AGE_MIN, max=AGE_MAX, step=1,
                    value=[AGE_MIN, AGE_MAX],
                    marks={},
                    tooltip=None,
                ),
            ], xs=12, sm=12, md=4),

            dbc.Col([
                html.Label("Metric", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="metric-radio",
                    options=[
                        {"label": "  Overall",   "value": "overall"},
                        {"label": "  Potential", "value": "potential"},
                    ],
                    value="overall", inline=True,
                    style={"color": TEXT, "fontSize": "13px", "marginTop": "6px"},
                    inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                ),
            ], xs=12, sm=6, md=2,
               className="d-flex flex-column justify-content-center"),
        ], align="center", className="g-3"),
    ]),
], style={**CARD_STYLE, "marginBottom": "28px"})

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    dbc.Row(dbc.Col(html.Div([
        html.H1("FIFA 23 Players Analytics Dashboard",
                style={"color": ACCENT, "fontWeight": "bold", "marginBottom": "6px"}),
        html.P("Exploring player ratings, market values, wages, and attributes "
               "across clubs, leagues, positions, and FIFA editions.",
               style={"color": TEXT, "fontSize": "15px", "marginBottom": "0"}),
    ], className="dashboard-header",
       style={"textAlign": "center", "padding": "30px 0 20px 0"}))),

    controls_panel,

    section_header("Section 1: Comparison Charts"),
    dbc.Row([
        dbc.Col(chart_card(
            "Which Club Has the Highest Average Rating? (Column Chart)",
            "col-chart", fig_col), md=6),
        dbc.Col(chart_card(
            "Top 15 Nationalities by Average Rating (Bar Chart)",
            "bar-chart", fig_bar), md=6),
    ]),

    section_header("Section 2: Stacked & Clustered Charts"),
    dbc.Row([
        dbc.Col(chart_card(
            "Player Composition by Position Group & Preferred Foot (Stacked Column)",
            "stacked-col-chart", fig_stacked_col, height=440), md=6),
        dbc.Col(chart_card(
            "Player Count Breakdown by Position Group per League (Stacked Bar)",
            "stacked-bar-chart", fig_stacked_bar, height=440), md=6),
    ]),
    dbc.Row([
        dbc.Col(chart_card(
            "Overall vs Potential by Position Group — Largest Gap? (Clustered Column)",
            "clustered-col-chart", fig_clust_col), md=6),
        dbc.Col(chart_card(
            "Average Weekly Wage by Position — Top 5 Leagues (Clustered Bar)",
            "clustered-bar-chart", fig_clust_bar), md=6),
    ]),

    section_header("Section 3: Relationship Charts"),
    dbc.Row([
        dbc.Col(chart_card(
            "Does Age Influence Overall Rating? (Scatter Chart)",
            "scatter-chart", fig_scatter), md=6),
        dbc.Col(chart_card(
            "Age vs Market Value — Bubble Size = Potential (Bubble Chart)",
            "bubble-chart", fig_bubble), md=6),
    ]),

    section_header("Section 4: Distribution Charts"),
    dbc.Row([
        dbc.Col(chart_card(
            "Overall Rating Frequency Distribution (Histogram)",
            "histogram-chart", fig_histogram, height=420), md=4),
        dbc.Col(chart_card(
            "Weekly Wage — Five-Number Summary (Box Chart)",
            "box-chart", fig_box, height=420), md=4),
        dbc.Col(chart_card(
            "Potential Distribution by Preferred Foot (Violin Chart)",
            "violin-chart", fig_violin, height=420), md=4),
    ]),

    section_header("Section 5: Time-Series Charts"),
    dbc.Row([
        dbc.Col(chart_card(
            "Average Rating by Age — Trend per Position Group (Line Chart)",
            "line-chart", fig_line), md=6),
        dbc.Col(chart_card(
            "Cumulative Player Count by Age (Area Chart)",
            "area-chart", fig_area), md=6),
    ]),

    html.Div(style={"height": "48px"}),

], fluid=True, style={"backgroundColor": BG, "minHeight": "100vh", "padding": "0 24px"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def _filter(league, age_range, position=None):
    dff = df.copy()
    if league != "ALL":
        dff = dff[dff["league_name"] == league]
    if position and position != "ALL":
        dff = dff[dff["position_group"] == position]
    if age_range:
        dff = dff[(dff["age"] >= age_range[0]) & (dff["age"] <= age_range[1])]
    return dff


# ── Callback 1: Comparison Charts (league + age + metric) ────────────────────
@app.callback(
    Output("col-chart", "figure"),
    Output("bar-chart", "figure"),
    Input("league-dropdown",  "value"),
    Input("age-slider",       "value"),
    Input("metric-radio",     "value"),
)
def update_comparison(league, age_range, metric):
    dff = _filter(league, age_range)
    if dff.empty:
        return empty_fig(), empty_fig()
    return build_col_chart(dff, metric), build_bar_chart(dff, metric)


# ── Callback 2: Stacked & Clustered Charts (league + position) ───────────────
@app.callback(
    Output("stacked-col-chart",  "figure"),
    Output("stacked-bar-chart",  "figure"),
    Output("clustered-col-chart","figure"),
    Output("clustered-bar-chart","figure"),
    Input("league-dropdown",   "value"),
    Input("position-dropdown", "value"),
)
def update_stacked_clustered(league, position):
    dff = _filter(league, age_range=None, position=position)
    if dff.empty:
        return empty_fig(), empty_fig(), empty_fig(), empty_fig()
    return (
        build_stacked_col_chart(dff),
        build_stacked_bar_chart(dff),
        build_clustered_col_chart(dff),
        build_clustered_bar_chart(dff),
    )


# ── Callback 3: Relationship Charts (position + age) ─────────────────────────
@app.callback(
    Output("scatter-chart", "figure"),
    Output("bubble-chart",  "figure"),
    Input("position-dropdown", "value"),
    Input("age-slider",        "value"),
)
def update_relationship(position, age_range):
    dff = _filter("ALL", age_range, position)
    if dff.empty:
        return empty_fig(), empty_fig()
    return build_scatter_chart(dff), build_bubble_chart(dff)


# ── Callback 4: Distribution Charts (position + age) ─────────────────────────
@app.callback(
    Output("histogram-chart", "figure"),
    Output("box-chart",       "figure"),
    Output("violin-chart",    "figure"),
    Input("position-dropdown", "value"),
    Input("age-slider",        "value"),
)
def update_distribution(position, age_range):
    dff = _filter("ALL", age_range, position)
    if dff.empty:
        return empty_fig(), empty_fig(), empty_fig()
    return build_histogram_chart(dff), build_box_chart(dff), build_violin_chart(dff)


# ── Callback 5: Time-Series Charts (league + metric) ─────────────────────────
@app.callback(
    Output("line-chart", "figure"),
    Output("area-chart", "figure"),
    Input("league-dropdown", "value"),
    Input("metric-radio",    "value"),
)
def update_timeseries(league, metric):
    dff = _filter(league, age_range=None)
    if dff.empty:
        return empty_fig(), empty_fig()
    return build_line_chart(dff, metric), build_area_chart(dff)


if __name__ == "__main__":
    app.run(debug=True)
