"""
Vehicle Telemetry Dashboard
----------------------------
Run with:
    pip install streamlit pandas plotly
    streamlit run vehicle_dashboard_app.py

Then open the local URL Streamlit prints (usually http://localhost:8501).
Or just double-click run_dashboard.bat if you're on Windows.

WHAT THIS APP DOES
- Auto-detects every CSV file in the same folder as this script (one per
  vehicle) and lets you switch between them, or upload a new one.
- Cleans glitch temperature sensor readings: any reading equal to
  ERROR_TEMP_VALUE (default -127) is replaced with the previous valid
  reading for that sensor (forward-fill), not blanked out.
- For each day:
    - lowest sht_temp and lowest ambient_temp, ONLY within working hours
      (8am-7pm local time)
    - "daily" total_distance and energy, RESET-ADJUSTED: total_distance and
      energy are cumulative counters that occasionally reset to zero
      mid-day. Every time a reset is detected, the value just before the
      reset is banked, and the day's total = sum of all banked pre-reset
      values + the value closest to 9pm in the final (still-open) segment.
    - distance is converted from meters to kilometers.
- For the month:
    - lowest sht_temp / ambient_temp across all days
    - total distance (km) = sum of each day's reset-adjusted total
    - total energy = sum of each day's reset-adjusted total
- Lets you pick a date and shows a chart of sht_temp & ambient_temp for that
  day (8am-7pm only), with lat/long/speed visible on hover for every point.
"""

import glob
import math
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import time, timedelta

# =============================================================================
# CONFIG — adjust these to match your setup
# =============================================================================
TIMEZONE_OFFSET_HOURS = 1                # UTC+1 = Africa/Lagos (WAT). Change if needed.
ERROR_TEMP_VALUE = -127                  # exact glitch value seen in the sensor logs
WORK_START = time(8, 0, 0)               # working-hours window start
WORK_END = time(19, 0, 0)                # working-hours window end (7pm)
NIGHT_TARGET = time(21, 0, 0)            # "9pm" reference point for distance/energy
RESET_EPSILON = 1e-6                     # tolerance below which a drop counts as a "reset"
DISTANCE_UNIT_DIVISOR = 1000.0           # source total_distance is in meters -> divide to get km

st.set_page_config(page_title="Vehicle Telemetry Dashboard", layout="wide")


# =============================================================================
# DATASET DISCOVERY
# =============================================================================
def discover_csv_files() -> list[str]:
    """Find every CSV sitting next to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(script_dir, "*.csv")))


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df["created_at_local"] = df["created_at"] + timedelta(hours=TIMEZONE_OFFSET_HOURS)
    df["date"] = df["created_at_local"].dt.date
    df["time_local"] = df["created_at_local"].dt.time

    # Replace glitch temperature readings with the previous valid reading
    # for that same sensor (forward-fill in chronological order). If the
    # very first reading(s) are glitches, there's no earlier value to use,
    # so fall back to the next valid reading instead.
    for col in ["sht_temp", "temp2", "ambient_temp"]:
        df[col] = df[col].where(df[col] != ERROR_TEMP_VALUE)
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()

    return df


def seconds_of(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def working_hours_mask(df: pd.DataFrame) -> pd.Series:
    return df["time_local"].apply(lambda t: WORK_START <= t <= WORK_END)


def closest_to_night(day_df: pd.DataFrame) -> pd.Series:
    """Return the record whose local time is closest to NIGHT_TARGET (9pm)."""
    target_s = seconds_of(NIGHT_TARGET)
    diffs = day_df["time_local"].apply(lambda t: abs(seconds_of(t) - target_s))
    return day_df.loc[diffs.idxmin()]


def reset_adjusted_total(day_df: pd.DataFrame, col: str, up_to_ts) -> float:
    """
    Sum a cumulative counter (distance/energy) for one day, correcting for
    mid-day resets. Every time the value decreases compared to the previous
    reading, that's a reset: bank the pre-reset value into `carry`. The
    final total is carry + the last raw reading at/before `up_to_ts`.
    """
    rows = day_df[day_df["created_at_local"] <= up_to_ts].sort_values("created_at_local")
    if rows.empty:
        return 0.0

    carry = 0.0
    prev = None
    for v in rows[col]:
        if prev is not None and v < prev - RESET_EPSILON:
            carry += prev
        prev = v

    final_val = rows[col].iloc[-1]
    return carry + final_val


# =============================================================================
# DAILY / MONTHLY SUMMARIES
# =============================================================================
@st.cache_data
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, day_df in df.groupby("date"):
        wh = day_df[working_hours_mask(day_df)]

        min_sht = wh["sht_temp"].min() if not wh["sht_temp"].dropna().empty else None
        min_ambient = wh["ambient_temp"].min() if not wh["ambient_temp"].dropna().empty else None

        night_row = closest_to_night(day_df)
        night_ts = night_row["created_at_local"]

        distance_total = reset_adjusted_total(day_df, "total_distance", night_ts) / DISTANCE_UNIT_DIVISOR
        energy_total = reset_adjusted_total(day_df, "energy", night_ts)

        rows.append({
            "date": day,
            "min_sht_temp": min_sht,
            "min_ambient_temp": min_ambient,
            "distance_km": distance_total,
            "energy": energy_total,
            "night_record_time": night_ts.strftime("%H:%M:%S"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_monthly_summary(daily: pd.DataFrame) -> dict:
    return {
        "min_sht_temp": daily["min_sht_temp"].min(),
        "min_sht_temp_date": daily.loc[daily["min_sht_temp"].idxmin(), "date"] if daily["min_sht_temp"].notna().any() else None,
        "min_ambient_temp": daily["min_ambient_temp"].min(),
        "min_ambient_temp_date": daily.loc[daily["min_ambient_temp"].idxmin(), "date"] if daily["min_ambient_temp"].notna().any() else None,
        "total_distance_km": daily["distance_km"].sum(),
        "total_energy": daily["energy"].sum(),
    }


# =============================================================================
# SIDEBAR — VEHICLE / DATASET SELECTION
# =============================================================================
st.sidebar.header("🚐 Vehicle dataset")

local_csvs = discover_csv_files()
local_labels = [os.path.basename(p) for p in local_csvs]

file_bytes = None
file_label = None

if local_csvs:
    picked_label = st.sidebar.selectbox("Choose a dataset found in this folder", local_labels)
    picked_path = local_csvs[local_labels.index(picked_label)]
    with open(picked_path, "rb") as f:
        file_bytes = f.read()
    file_label = picked_label

st.sidebar.markdown("— or —")
uploaded = st.sidebar.file_uploader("Upload a different CSV", type=["csv"])
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_label = uploaded.name

if file_bytes is None:
    st.warning("No CSV files found next to this script, and nothing uploaded yet. "
               "Place your vehicle CSV file(s) in the same folder as this script, or upload one from the sidebar.")
    st.stop()

st.sidebar.caption(f"Loaded: {file_label}")


# =============================================================================
# APP
# =============================================================================
df = load_data(file_bytes)
daily = build_daily_summary(df)
monthly = build_monthly_summary(daily)

plate = df["vehicle_plate_number"].iloc[0] if "vehicle_plate_number" in df.columns else file_label

st.title("🚐 Vehicle Telemetry Dashboard")
st.caption(f"Vehicle: **{plate}** — timestamps shown in local time (UTC+{TIMEZONE_OFFSET_HOURS}). "
           f"Working hours = {WORK_START.strftime('%H:%M')}–{WORK_END.strftime('%H:%M')}.")

st.subheader("Monthly Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lowest sht_temp (month)",
          f"{monthly['min_sht_temp']:.1f} °C" if monthly['min_sht_temp'] is not None else "N/A",
          help=f"On {monthly['min_sht_temp_date']}" if monthly['min_sht_temp_date'] else None)
c2.metric("Lowest ambient_temp (month)",
          f"{monthly['min_ambient_temp']:.1f} °C" if monthly['min_ambient_temp'] is not None else "N/A",
          help=f"On {monthly['min_ambient_temp_date']}" if monthly['min_ambient_temp_date'] else None)
c3.metric("Total distance (month)", f"{monthly['total_distance_km']:.2f} km")
c4.metric("Total energy (month)", f"{monthly['total_energy']:.4f}")

st.divider()

# --- Date picker ---
available_dates = sorted(daily["date"].unique())
selected_date = st.selectbox(
    "📅 Select a date to inspect",
    options=available_dates,
    format_func=lambda d: d.strftime("%A, %d %B %Y"),
    index=len(available_dates) - 1,
)

day_row = daily[daily["date"] == selected_date].iloc[0]
day_df = df[df["date"] == selected_date]
day_wh = day_df[working_hours_mask(day_df)].sort_values("created_at_local")

st.subheader(f"Day summary — {selected_date.strftime('%A, %d %B %Y')}")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Lowest sht_temp (working hrs)",
          f"{day_row['min_sht_temp']:.1f} °C" if pd.notna(day_row['min_sht_temp']) else "N/A")
d2.metric("Lowest ambient_temp (working hrs)",
          f"{day_row['min_ambient_temp']:.1f} °C" if pd.notna(day_row['min_ambient_temp']) else "N/A")
d3.metric("Distance (reset-adjusted)", f"{day_row['distance_km']:.2f} km")
d4.metric("Energy (reset-adjusted)", f"{day_row['energy']:.4f}")
st.caption(f"Reset-adjusted totals sum every pre-reset high point up to the record closest to "
           f"{NIGHT_TARGET.strftime('%H:%M')} local time ({day_row['night_record_time']} was used for this day).")

# --- Chart: sht_temp & ambient_temp, working hours only, hover shows lat/long/speed ---
if day_wh.empty:
    st.warning("No records found in the working-hours window (8am–7pm) for this date.")
else:
    fig = go.Figure()

    hover_template = (
        "Time: %{x|%H:%M:%S}<br>"
        "Temp: %{y:.1f} °C<br>"
        "Lat: %{customdata[0]:.5f}<br>"
        "Lng: %{customdata[1]:.5f}<br>"
        "Speed: %{customdata[2]:.1f}"
        "<extra>%{fullData.name}</extra>"
    )

    fig.add_trace(go.Scatter(
        x=day_wh["created_at_local"],
        y=day_wh["sht_temp"],
        mode="lines+markers",
        name="sht_temp",
        customdata=day_wh[["lat", "long", "speed"]].values,
        hovertemplate=hover_template,
    ))

    fig.add_trace(go.Scatter(
        x=day_wh["created_at_local"],
        y=day_wh["ambient_temp"],
        mode="lines+markers",
        name="ambient_temp",
        customdata=day_wh[["lat", "long", "speed"]].values,
        hovertemplate=hover_template,
    ))

    fig.update_layout(
        xaxis_title="Time (8am–7pm)",
        yaxis_title="Temperature (°C)",
        legend_title="Sensor",
        hovermode="closest",
        height=500,
    )

    st.plotly_chart(fig, width='stretch')

with st.expander("View raw records for this day (working hours)"):
    st.dataframe(day_wh[["created_at_local", "sht_temp", "ambient_temp", "lat", "long", "speed"]])

st.divider()
st.subheader("Full month — daily breakdown")
st.dataframe(
    daily.rename(columns={
        "date": "Date",
        "min_sht_temp": "Min sht_temp (°C)",
        "min_ambient_temp": "Min ambient_temp (°C)",
        "distance_km": "Distance (km, reset-adjusted)",
        "energy": "Energy (reset-adjusted)",
        "night_record_time": "Reference record time",
    }),
    width='stretch',
)


# =============================================================================
# LOCATION HEAT MAP (free OpenStreetMap tiles, no API key needed)
# =============================================================================
def estimate_zoom(lat_span: float, lon_span: float) -> float:
    """Rough zoom level so all points fit in a ~900px wide map."""
    span = max(lat_span, lon_span, 1e-4)
    return float(min(max(math.log2(900 * 360 / (256 * span)) - 0.3, 2), 17))


st.divider()
st.subheader("📍 Location heat map")

m1, m2, m3 = st.columns(3)
map_scope = m1.radio("Show data for", ["Selected day", "Full month"], horizontal=True)
map_wh_only = m2.checkbox("Working hours only (8am–7pm)", value=True)
map_moving_only = m3.checkbox("Exclude stationary points (speed = 0)", value=False)
heat_radius = st.slider("Heat radius", min_value=5, max_value=60, value=20)

map_df = df if map_scope == "Full month" else day_df
if map_wh_only:
    map_df = map_df[working_hours_mask(map_df)]
if map_moving_only:
    map_df = map_df[map_df["speed"] > 0]
map_df = map_df.dropna(subset=["lat", "long"])
map_df = map_df[(map_df["lat"] != 0) & (map_df["long"] != 0)]  # drop GPS "no fix" points

if map_df.empty:
    st.info("No GPS points match the current filters.")
else:
    center_lat = float(map_df["lat"].mean())
    center_lon = float(map_df["long"].mean())
    zoom = estimate_zoom(map_df["lat"].max() - map_df["lat"].min(),
                         map_df["long"].max() - map_df["long"].min())

    try:
        heat_fig = px.density_map(
            map_df, lat="lat", lon="long", radius=heat_radius,
            center=dict(lat=center_lat, lon=center_lon), zoom=zoom,
            map_style="open-street-map", color_continuous_scale=["rgba(0,0,255,0)", "#7ec8ff", "#1e6fff", "#0033a0"]",
            height=600,
        )
    except AttributeError:  # older plotly versions
        heat_fig = px.density_mapbox(
            map_df, lat="lat", lon="long", radius=heat_radius,
            center=dict(lat=center_lat, lon=center_lon), zoom=zoom,
            mapbox_style="open-street-map", color_continuous_scale="YlOrRd",
            height=600,
        )

    heat_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
    st.plotly_chart(heat_fig, width='stretch')
    st.caption(f"{len(map_df):,} GPS points plotted. Brighter areas = more readings recorded there "
               f"(so places where the vehicle spent more time). Map © OpenStreetMap contributors.")
