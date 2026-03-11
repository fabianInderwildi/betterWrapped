"""
betterWrapped – Dash Plotly Dashboard
Two interactive tabs:
  Tab 1 – Static Data  : "Your Favorites"        (bar chart, top-N)
  Tab 2 – Dynamic Data : "Filtered Items Over Time" (line / bar chart)

Run:
    pip install dash plotly
    python dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import os
import pandas as pd
import numpy as np
import pycountry
from dotenv import load_dotenv

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants / colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR      = "#121212"
SURFACE_COLOR = "#1e1e1e"
CARD_COLOR    = "#282828"
TEXT_COLOR    = "#FFFFFF"
SUBTEXT_COLOR = "#B3B3B3"
ACCENT_COLOR  = "#1DB954"          # Spotify green
GREY_SCALE    = [
    "#000000", "#1a1a1a", "#333333", "#4d4d4d", "#666666",
    "#7f7f7f", "#999999", "#B3B3B3", "#CCCCCC", "#E6E6E6", "#FFFFFF",
]

dict_att_disp = {
    "artist_name": "Artist",  "track_name": "Track",  "genre": "Genre",
    "album_name":  "Album",   "country_name": "Country",
    "region_name": "Region",  "city_name": "City",    "platform": "Platform",
    "ms_played":   "Milliseconds", "s_played": "Seconds",
    "min_played":  "Minutes",      "h_played": "Hours",
    "d_played":    "Days",         "count":    "Count",
}

month_names = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
weekday_names = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_streaming_history() -> pd.DataFrame:
    folder = os.path.join(BASE_DIR, "Spotify Extended Streaming History")
    frames = [
        pd.read_json(os.path.join(folder, f))
        for f in sorted(os.listdir(folder))
        if f.startswith("Streaming_History_Audio")
    ]
    return pd.concat(frames, ignore_index=True)


df = _load_streaming_history()

df.drop(
    columns=[
        "episode_name", "episode_show_name", "spotify_episode_uri",
        "audiobook_title", "audiobook_uri",
        "audiobook_chapter_uri", "audiobook_chapter_title",
    ],
    errors="ignore",
    inplace=True,
)
df.rename(
    columns={
        "conn_country":                        "country_code",
        "ip_addr":                             "ip",
        "master_metadata_track_name":          "track_name",
        "master_metadata_album_artist_name":   "artist_name",
        "master_metadata_album_album_name":    "album_name",
    },
    inplace=True,
)

df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
df.dropna(subset=["artist_name"], inplace=True)

df["s_played"]   = df["ms_played"] / 1_000
df["min_played"] = df["ms_played"] / 60_000
df["h_played"]   = df["ms_played"] / 3_600_000
df["d_played"]   = df["ms_played"] / 86_400_000

df["country_name"] = df["country_code"].map(
    lambda x: pycountry.countries.get(alpha_2=x).name
    if pycountry.countries.get(alpha_2=x) else "Unknown"
)
df["date"]  = df["ts"].dt.date
df["year"]  = df["ts"].dt.year
df["month"] = df["ts"].dt.month

if "spotify_track_uri" in df.columns:
    df["spotify_track_uri"] = df["spotify_track_uri"].str.split(":").str[-1]

# Merge genre data
_genre_path = os.path.join(BASE_DIR, "fetched_data", "artist_genre.csv")
if os.path.exists(_genre_path):
    df = df.merge(pd.read_csv(_genre_path), on="artist_name", how="left")
else:
    df["genre"] = pd.NA

# Merge IP-location data
_ip_path = os.path.join(BASE_DIR, "fetched_data", "ip_loc.csv")
if os.path.exists(_ip_path):
    df_ip = pd.read_csv(_ip_path)
    df = df.merge(df_ip, on="ip", how="left")
    if "geo_loc" in df.columns:
        _split = df["geo_loc"].str.split(",", expand=True)
        df["lat"] = pd.to_numeric(_split[0], errors="coerce")
        df["lon"] = pd.to_numeric(_split[1], errors="coerce")

# Ensure optional location columns always exist
for _col in ("region_name", "city_name"):
    if _col not in df.columns:
        df[_col] = pd.NA

TS_MIN = df["ts"].min()
TS_MAX = df["ts"].max()

# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_top_data(
    attribute: str,
    sorting_value: str,
    time_format: str,
    top_count: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (df["ts"] > start) & (df["ts"] < end)
    sub  = df[mask]

    if sorting_value == "track_count":
        if attribute == "genre":
            return (
                sub["genre"].str.split(", ").explode()
                .value_counts().head(top_count).reset_index()
            )
        return sub[attribute].value_counts().head(top_count).reset_index()

    # sorting_value == "listening_time"
    if attribute == "genre":
        tmp = sub[["genre", time_format]].copy()
        tmp["genre"] = tmp["genre"].str.split(", ")
        return (
            tmp.explode("genre")
            .groupby("genre")[time_format].sum()
            .sort_values(ascending=False).head(top_count)
            .reset_index().round(1)
        )
    return (
        sub[[attribute, time_format]]
        .groupby(attribute).sum()
        .sort_values(time_format, ascending=False)
        .head(top_count).reset_index().round(1)
    )


def get_listening_dates_counted(
    attribute: str,
    item: list,
    measuring_value: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fmt: str,
) -> pd.DataFrame:
    mask = (df["ts"] > start) & (df["ts"] < end)
    cols = [
        attribute, "date",
        "ms_played", "s_played", "min_played", "h_played", "d_played",
    ]

    if item == ["all"]:
        df_fi = df[mask][cols]
    elif attribute == "genre":
        genre_mask = (
            df["genre"].fillna("").str.split(r",\s*")
            .apply(lambda gs: any(t.strip() == g.strip() for g in gs for t in item))
        )
        df_fi = df[mask & genre_mask][cols]
    else:
        df_fi = df[mask & df[attribute].isin(item)][cols]

    # Aggregate listening time per day
    df_lt = (
        df_fi.groupby("date")[["ms_played","s_played","min_played","h_played","d_played"]]
        .sum().reset_index()
    )
    # Count plays per day
    df_tc = df_fi.groupby("date").size().reset_index(name="count")

    df_dates = df_lt.merge(df_tc, on="date", how="left")
    df_dates["date"]  = pd.to_datetime(df_dates["date"])
    df_dates["count"] = df_dates["count"].fillna(0).astype(int)

    # Fill every calendar day between start and end, default 0
    full_range = pd.date_range(start=start, end=end, freq="d")
    df_filled = (
        pd.DataFrame({"date": full_range})
        .merge(df_dates, on="date", how="left")
        .sort_values("date")
        .fillna(0)
    )

    n_years = max(1, TS_MAX.year - TS_MIN.year)
    n_weeks = n_years * 52

    if fmt == "average_weekday":
        res = (
            df_filled.groupby(df_filled["date"].dt.weekday)[measuring_value]
            .sum() / n_weeks
        ).reset_index()
        res["date"] = res["date"].map(weekday_names)
        return res

    if fmt == "average_month":
        res = (
            df_filled.groupby(df_filled["date"].dt.month)[measuring_value]
            .sum() / n_years
        ).reset_index()
        res["date"] = res["date"].map(month_names)
        return res

    if fmt == "day":
        return df_filled[["date", measuring_value]]

    if fmt == "month":
        res = (
            df_filled.groupby(df_filled["date"].dt.to_period("M"))[measuring_value]
            .sum().reset_index()
        )
        res["date"] = pd.to_datetime(res["date"].astype(str))
        return res

    # fmt == "year"
    return (
        df_filled.groupby(df_filled["date"].dt.year)[measuring_value]
        .sum().reset_index()
    )


def get_attribute_options(attribute: str) -> list[str]:
    """Return sorted unique values for the given attribute column."""
    if attribute == "genre":
        vals = df["genre"].dropna().str.split(", ").explode().str.strip().unique()
    else:
        vals = df[attribute].dropna().unique()
    return sorted(str(v) for v in vals if str(v))


# ─────────────────────────────────────────────────────────────────────────────
# Plotly base layout factory
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(title_text: str = "") -> dict:
    return dict(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),
        title=dict(text=title_text, font=dict(color=TEXT_COLOR, size=18), x=0.5),
        xaxis=dict(
            gridcolor="#3a3a3a", linecolor="#555",
            tickfont=dict(color=SUBTEXT_COLOR),
            title_font=dict(color=SUBTEXT_COLOR),
        ),
        yaxis=dict(
            gridcolor="#3a3a3a", linecolor="#555",
            tickfont=dict(color=SUBTEXT_COLOR),
            title_font=dict(color=SUBTEXT_COLOR),
        ),
        margin=dict(l=65, r=50, t=100, b=90),
        hoverlabel=dict(bgcolor=SURFACE_COLOR, font_color=TEXT_COLOR),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Option lists for dropdowns
# ─────────────────────────────────────────────────────────────────────────────
ATTRIBUTE_OPTIONS = [
    {"label": "Artist",   "value": "artist_name"},
    {"label": "Track",    "value": "track_name"},
    {"label": "Genre",    "value": "genre"},
    {"label": "Album",    "value": "album_name"},
    {"label": "Country",  "value": "country_name"},
    {"label": "Region",   "value": "region_name"},
    {"label": "City",     "value": "city_name"},
    {"label": "Platform", "value": "platform"},
]

TIME_FORMAT_OPTIONS = [
    {"label": "Milliseconds", "value": "ms_played"},
    {"label": "Seconds",      "value": "s_played"},
    {"label": "Minutes",      "value": "min_played"},
    {"label": "Hours",        "value": "h_played"},
    {"label": "Days",         "value": "d_played"},
]

MEASURING_VALUE_OPTIONS = [
    {"label": "Count",        "value": "count"},
    {"label": "Milliseconds", "value": "ms_played"},
    {"label": "Seconds",      "value": "s_played"},
    {"label": "Minutes",      "value": "min_played"},
    {"label": "Hours",        "value": "h_played"},
    {"label": "Days",         "value": "d_played"},
]

FORMAT_OPTIONS = [
    {"label": "Average Weekday", "value": "average_weekday"},
    {"label": "Average Month",   "value": "average_month"},
    {"label": "Day",             "value": "day"},
    {"label": "Month",           "value": "month"},
    {"label": "Year",            "value": "year"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Reusable CSS snippets
# ─────────────────────────────────────────────────────────────────────────────
_lbl = {
    "color": SUBTEXT_COLOR, "fontSize": "11px", "marginBottom": "5px",
    "display": "block", "letterSpacing": "0.07em", "textTransform": "uppercase",
}
_sec  = {"marginBottom": "22px"}
_ctrl = {
    "width": "280px", "minWidth": "240px", "flexShrink": "0",
    "background": CARD_COLOR, "padding": "20px 18px", "borderRadius": "8px",
    "overflowY": "auto", "maxHeight": "calc(100vh - 150px)",
}
_gfx  = {
    "flex": "1", "background": CARD_COLOR, "borderRadius": "8px", "padding": "8px",
}
_row = {
    "display": "flex", "gap": "16px", "padding": "16px",
    "minHeight": "calc(100vh - 150px)",
}
_heading = {
    "color": TEXT_COLOR, "margin": "0 0 20px", "fontSize": "13px",
    "letterSpacing": "0.1em", "textTransform": "uppercase", "fontWeight": "600",
}
_tab_off = {
    "background": SURFACE_COLOR, "color": SUBTEXT_COLOR,
    "padding": "10px 22px", "fontFamily": "Arial, sans-serif",
}
_tab_on = {
    "background": BG_COLOR, "color": TEXT_COLOR,
    "padding": "10px 22px", "fontFamily": "Arial, sans-serif",
    "borderTop": f"2px solid {ACCENT_COLOR}",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="betterWrapped")
server = app.server   # expose for WSGI deployment

app.layout = html.Div(
    style={"background": BG_COLOR, "minHeight": "100vh", "fontFamily": "Arial, sans-serif"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": SURFACE_COLOR, "padding": "14px 24px",
                "display": "flex", "alignItems": "center", "gap": "12px",
            },
            children=[
                html.Span("◉", style={"color": ACCENT_COLOR, "fontSize": "26px", "lineHeight": "1"}),
                html.H1(
                    "betterWrapped",
                    style={"color": TEXT_COLOR, "margin": "0", "fontSize": "22px", "fontWeight": "700"},
                ),
                html.Span(
                    f"{TS_MIN.strftime('%d.%m.%Y')} – {TS_MAX.strftime('%d.%m.%Y')}",
                    style={"color": SUBTEXT_COLOR, "fontSize": "13px", "marginLeft": "auto"},
                ),
            ],
        ),

        # ── Tabs ──────────────────────────────────────────────────────────────
        dcc.Tabs(
            value="tab-static",
            colors={"border": SURFACE_COLOR, "primary": ACCENT_COLOR, "background": SURFACE_COLOR},
            children=[

                # ════════════════════════════════════════════════════════════
                # Tab 1 – Static Data │ Your Favorites
                # ════════════════════════════════════════════════════════════
                dcc.Tab(
                    label="Your Favorites",
                    value="tab-static",
                    style=_tab_off,
                    selected_style=_tab_on,
                    children=[
                        html.Div(
                            style=_row,
                            children=[

                                # ── Control panel ─────────────────────────
                                html.Div(
                                    style=_ctrl,
                                    children=[
                                        html.H3("Parameters", style=_heading),

                                        # Attribute
                                        html.Div(style=_sec, children=[
                                            html.Label("Attribute", style=_lbl),
                                            dcc.Dropdown(
                                                id="fav-attribute",
                                                options=ATTRIBUTE_OPTIONS,
                                                value="genre",
                                                clearable=False,
                                            ),
                                        ]),

                                        # Top Count
                                        html.Div(style=_sec, children=[
                                            html.Label("Top Count", style=_lbl),
                                            dcc.Slider(
                                                id="fav-top-count",
                                                min=1, max=25, step=1, value=10,
                                                marks={1: "1", 5: "5", 10: "10",
                                                       15: "15", 20: "20", 25: "25"},
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                                        ]),

                                        # Sorting
                                        html.Div(style=_sec, children=[
                                            html.Label("Sorting", style=_lbl),
                                            dcc.RadioItems(
                                                id="fav-sorting",
                                                options=[
                                                    {"label": " Listening Time", "value": "listening_time"},
                                                    {"label": " Track Count",    "value": "track_count"},
                                                ],
                                                value="listening_time",
                                                labelStyle={
                                                    "color": TEXT_COLOR, "fontSize": "13px",
                                                    "display": "block", "marginBottom": "7px",
                                                },
                                                inputStyle={"marginRight": "8px", "accentColor": ACCENT_COLOR},
                                            ),
                                        ]),

                                        # Time Format
                                        html.Div(style=_sec, children=[
                                            html.Label("Time Format", style=_lbl),
                                            dcc.Dropdown(
                                                id="fav-time-format",
                                                options=TIME_FORMAT_OPTIONS,
                                                value="h_played",
                                                clearable=False,
                                            ),
                                        ]),

                                        # Date Range
                                        html.Div(style=_sec, children=[
                                            html.Label("Date Range", style=_lbl),
                                            dcc.DatePickerRange(
                                                id="fav-date-range",
                                                min_date_allowed=TS_MIN.date(),
                                                max_date_allowed=TS_MAX.date(),
                                                start_date=TS_MIN.date(),
                                                end_date=TS_MAX.date(),
                                                display_format="DD.MM.YYYY",
                                                style={"width": "100%"},
                                            ),
                                        ]),
                                    ],
                                ),

                                # ── Chart ─────────────────────────────────
                                html.Div(
                                    style=_gfx,
                                    children=[
                                        dcc.Graph(
                                            id="fav-graph",
                                            style={"height": "calc(100vh - 175px)"},
                                            config={"displayModeBar": True, "displaylogo": False},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # ════════════════════════════════════════════════════════════
                # Tab 2 – Dynamic Data │ Filtered Items Over Time
                # ════════════════════════════════════════════════════════════
                dcc.Tab(
                    label="Filtered Items Over Time",
                    value="tab-dynamic",
                    style=_tab_off,
                    selected_style=_tab_on,
                    children=[
                        html.Div(
                            style=_row,
                            children=[

                                # ── Control panel ─────────────────────────
                                html.Div(
                                    style=_ctrl,
                                    children=[
                                        html.H3("Parameters", style=_heading),

                                        # Attribute
                                        html.Div(style=_sec, children=[
                                            html.Label("Attribute", style=_lbl),
                                            dcc.Dropdown(
                                                id="filt-attribute",
                                                options=ATTRIBUTE_OPTIONS,
                                                value="genre",
                                                clearable=False,
                                            ),
                                        ]),

                                        # Item selector  (multi-select Dropdown; empty = all)
                                        html.Div(style=_sec, children=[
                                            html.Label(
                                                [
                                                    "Items  ",
                                                    html.Span(
                                                        "(empty = all)",
                                                        style={
                                                            "fontStyle": "italic", "fontSize": "10px",
                                                            "color": GREY_SCALE[6], "textTransform": "none",
                                                        },
                                                    ),
                                                ],
                                                style=_lbl,
                                            ),
                                            dcc.Dropdown(
                                                id="filt-item",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="All items (no filter)",
                                                searchable=True,
                                            ),
                                        ]),

                                        # Measuring Value
                                        html.Div(style=_sec, children=[
                                            html.Label("Measuring Value", style=_lbl),
                                            dcc.Dropdown(
                                                id="filt-measuring-value",
                                                options=MEASURING_VALUE_OPTIONS,
                                                value="h_played",
                                                clearable=False,
                                            ),
                                        ]),

                                        # Date Range
                                        html.Div(style=_sec, children=[
                                            html.Label("Date Range", style=_lbl),
                                            dcc.DatePickerRange(
                                                id="filt-date-range",
                                                min_date_allowed=TS_MIN.date(),
                                                max_date_allowed=TS_MAX.date(),
                                                start_date=TS_MIN.date(),
                                                end_date=TS_MAX.date(),
                                                display_format="DD.MM.YYYY",
                                                style={"width": "100%"},
                                            ),
                                        ]),

                                        # Format
                                        html.Div(style=_sec, children=[
                                            html.Label("Format", style=_lbl),
                                            dcc.Dropdown(
                                                id="filt-format",
                                                options=FORMAT_OPTIONS,
                                                value="year",
                                                clearable=False,
                                            ),
                                        ]),
                                    ],
                                ),

                                # ── Chart ─────────────────────────────────
                                html.Div(
                                    style=_gfx,
                                    children=[
                                        dcc.Graph(
                                            id="filt-graph",
                                            style={"height": "calc(100vh - 175px)"},
                                            config={"displayModeBar": True, "displaylogo": False},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("fav-graph", "figure"),
    Input("fav-attribute",   "value"),
    Input("fav-top-count",   "value"),
    Input("fav-sorting",     "value"),
    Input("fav-time-format", "value"),
    Input("fav-date-range",  "start_date"),
    Input("fav-date-range",  "end_date"),
)
def update_favorites(attribute, top_count, sorting_value, time_format,
                     start_date, end_date):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    df_top     = get_top_data(attribute, sorting_value, time_format, top_count, start, end)
    categories = df_top[attribute].astype(str).tolist()
    values     = df_top.iloc[:, -1].tolist()

    x_label = dict_att_disp.get(attribute, attribute)
    y_label = (
        dict_att_disp.get(time_format, time_format)
        if sorting_value == "listening_time" else "Play Count"
    )
    title = (
        f"Top {top_count} {x_label}s  ·  "
        f"{sorting_value.replace('_', ' ').title()}<br>"
        f"<sup>{start.strftime('%d.%m.%Y')} – {end.strftime('%d.%m.%Y')}</sup>"
    )

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=ACCENT_COLOR,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in values],
            textposition="outside",
            textfont=dict(color=SUBTEXT_COLOR, size=11),
            hovertemplate=f"<b>%{{x}}</b><br>{y_label}: <b>%{{y:.2f}}</b><extra></extra>",
        )
    )
    fig.update_layout(
        **base_layout(title),
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_tickangle=-35,
        bargap=0.35,
    )
    return fig


@app.callback(
    Output("filt-item", "options"),
    Output("filt-item", "value"),
    Input("filt-attribute", "value"),
    State("filt-item",      "value"),
)
def refresh_item_options(attribute, current_values):
    """Repopulate the items dropdown when the attribute changes.
    Items that are no longer valid for the new attribute are removed."""
    options   = get_attribute_options(attribute)
    option_set = set(options)
    valid = [v for v in (current_values or []) if v in option_set]
    return [{"label": v, "value": v} for v in options], valid


@app.callback(
    Output("filt-graph", "figure"),
    Input("filt-attribute",      "value"),
    Input("filt-item",           "value"),
    Input("filt-measuring-value","value"),
    Input("filt-date-range",     "start_date"),
    Input("filt-date-range",     "end_date"),
    Input("filt-format",         "value"),
)
def update_filtered(attribute, item, measuring_value, start_date, end_date, fmt):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    items = item if item else ["all"]

    df_res = get_listening_dates_counted(attribute, items, measuring_value, start, end, fmt)
    x      = df_res["date"].astype(str).tolist()
    y      = df_res[measuring_value].tolist()

    if items == ["all"]:
        item_label = "All"
    elif len(items) <= 2:
        item_label = " & ".join(items)
    else:
        item_label = f"{', '.join(items[:2])} +{len(items) - 2} more"

    y_label = dict_att_disp.get(measuring_value, measuring_value)
    title = (
        f"{dict_att_disp.get(attribute, attribute)}:  {item_label}  ·  "
        f"{fmt.replace('_', ' ').title()}<br>"
        f"<sup>{start.strftime('%d.%m.%Y')} – {end.strftime('%d.%m.%Y')}</sup>"
    )

    categorical = fmt in ("average_weekday", "average_month")
    fig = go.Figure()

    if categorical:
        fig.add_trace(
            go.Bar(
                x=x, y=y,
                marker_color=ACCENT_COLOR,
                marker_line_color="rgba(0,0,0,0)",
                hovertemplate=f"<b>%{{x}}</b><br>{y_label}: <b>%{{y:.2f}}</b><extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines+markers",
                line=dict(color=ACCENT_COLOR, width=2),
                marker=dict(color=ACCENT_COLOR, size=4),
                hovertemplate=f"<b>%{{x}}</b><br>{y_label}: <b>%{{y:.2f}}</b><extra></extra>",
            )
        )

    fig.update_layout(
        **base_layout(title),
        xaxis_title=fmt.replace("_", " ").title(),
        yaxis_title=y_label,
        xaxis_tickangle=-35,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
