from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):

  class _RemainderColsList(list):
    pass

  _ct._RemainderColsList = _RemainderColsList

# -----------------------------------------------------------------------------
# Page Config & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Fast ETA Predictor | AI Lab",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .hero-card {
        background: linear-gradient(135deg, #e23744 0%, #b91c1c 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px rgba(226, 55, 68, 0.25);
    }
    .hero-card h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        color: white;
    }
    .hero-card p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }
    .eta-box {
        background: #0f172a;
        border: 2px solid #e23744;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        color: #f8fafc;
    }
    .eta-value {
        font-size: 3.1rem;
        font-weight: 800;
        color: #fb7185;
        line-height: 1.1;
        margin: 0.4rem 0;
    }
    .eta-sub {
        color: #94a3b8;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Constants & Defaults from zomato_fast_eta_metadata.json
# -----------------------------------------------------------------------------
MODEL_FILE = Path("zomato_fast_eta_model.joblib")
META_FILE = Path("zomato_fast_eta_metadata.json")

DEFAULT_METADATA = {
    "project": "Zomato Fast ETA Predictor",
    "model_type": "Random Forest Regressor",
    "target": "Time_taken (min)",
    "distance_method": "Haversine",
    "features": [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "distance_km",
        "order_hour",
        "order_day_of_week",
        "GPS_valid",
        "distance_valid",
        "order_time_missing",
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
    ],
    "categorical_features": [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
    ],
    "hyperparameters": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": 0.5,
        "random_state": 42,
    },
    "validation_metrics": {
        "rmse_minutes": 4.058,
        "mae_minutes": 3.219,
        "r2": 0.817,
        "within_5_minutes_percent": 80.26,
    },
    "test_metrics": {
        "rmse_minutes": 4.032,
        "mae_minutes": 3.214,
        "r2": 0.811,
        "within_5_minutes_percent": 80.38,
    },
    "notes": [
        "Chronological train/validation/test split was used.",
        "Time_Order_picked was excluded because it occurs after order placement.",
        "Haversine distance is the production distance feature.",
        (
            "Road-network distance was investigated separately and is not part"
            " of this model."
        ),
    ],
    "python_version": "3.13.16",
    "pandas_version": "2.2.3",
    "numpy_version": "2.1.3",
    "scikit_learn_version": "1.6.1",
}

DEFAULT_CATEGORIES = {
    "Weather_conditions": [
        "Sunny",
        "Cloudy",
        "Fog",
        "Stormy",
        "Sandstorms",
        "Windy",
    ],
    "Road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "Type_of_order": ["Snack", "Meal", "Drinks", "Buffet"],
    "Type_of_vehicle": [
        "motorcycle",
        "scooter",
        "electric_scooter",
        "bicycle",
    ],
    "Festival": ["No", "Yes"],
    "City": ["Metropolitian", "Urban", "Semi-Urban"],
}

ROUTE_PRESETS = {
    "Chandigarh Sector 17 ➔ Kharar / CU": (30.7398, 76.7827, 30.7463, 76.6469),
    "Mohali Phase 7 ➔ Chandigarh Sector 35": (
        30.7130,
        76.7262,
        30.7233,
        76.7601,
    ),
    "New Delhi (Connaught Place ➔ Hauz Khas)": (
        28.6315,
        77.2167,
        28.5494,
        77.2001,
    ),
    "Bengaluru (Indiranagar ➔ Koramangala)": (
        12.9784,
        77.6408,
        12.9352,
        77.6245,
    ),
    "Mumbai (Bandra West ➔ Andheri East)": (19.0596, 72.8295, 19.1136, 72.8697),
    "Custom Coordinates": None,
}

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Compute great-circle Haversine distance in kilometers."""
  r = 6371.0
  phi1, phi2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlambda = math.radians(lon2 - lon1)
  a = (
      math.sin(dphi / 2.0) ** 2
      + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
  )
  return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


@st.cache_resource(show_spinner="Loading Random Forest model...")
def load_local_model(path_str: str):
  return joblib.load(path_str)


def load_metadata() -> dict:
  if META_FILE.exists():
    try:
      return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
      pass
  return DEFAULT_METADATA


def extract_pipeline_categories(model) -> dict:
  """Inspect fitted sklearn pipeline to pull exact category strings if available."""
  cats = {k: list(v) for k, v in DEFAULT_CATEGORIES.items()}
  if model is None:
    return cats
  try:
    steps = getattr(model, "named_steps", {})
    for step in steps.values():
      transformers = getattr(step, "transformers_", [])
      for _, trans, cols in transformers:
        enc = trans
        if hasattr(trans, "named_steps"):
          for sub in trans.named_steps.values():
            if hasattr(sub, "categories_"):
              enc = sub
              break
        if hasattr(enc, "categories_"):
          for col, cat_arr in zip(cols, enc.categories_):
            cleaned = [
                str(x)
                for x in cat_arr
                if x is not None and str(x).lower() != "nan"
            ]
            if cleaned:
              cats[col] = cleaned
  except Exception:
    pass
  return cats


def extract_feature_importances(model) -> pd.DataFrame | None:
  """Extract feature importances from the final estimator or pipeline."""
  if model is None:
    return None
  try:
    estimator = model
    preprocessor = None
    if hasattr(model, "named_steps"):
      steps = list(model.named_steps.values())
      estimator = steps[-1]
      if len(steps) > 1:
        preprocessor = steps[0]

    if not hasattr(estimator, "feature_importances_"):
      return None

    importances = estimator.feature_importances_
    if preprocessor is not None and hasattr(
        preprocessor, "get_feature_names_out"
    ):
      names = [
          str(n).split("__")[-1] for n in preprocessor.get_feature_names_out()
      ]
    elif hasattr(model, "feature_names_in_"):
      names = list(model.feature_names_in_)
    else:
      names = [f"Feature {i}" for i in range(len(importances))]

    if len(names) != len(importances):
      names = [f"Feature {i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"Feature": names, "Importance": importances})
    return (
        df_imp.sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
  except Exception:
    return None


def build_feature_row(
    age: int,
    ratings: float,
    veh_cond: int,
    multi_del: int,
    rest_lat: float,
    rest_lon: float,
    del_lat: float,
    del_lon: float,
    dist_km: float,
    order_hour: int,
    order_dow: int,
    gps_valid: int,
    dist_valid: int,
    time_missing: int,
    weather: str,
    traffic: str,
    order_type: str,
    veh_type: str,
    festival: str,
    city: str,
) -> dict:
  return {
      "Delivery_person_Age": float(age),
      "Delivery_person_Ratings": float(ratings),
      "Vehicle_condition": int(veh_cond),
      "multiple_deliveries": float(multi_del),
      "Restaurant_latitude": float(rest_lat),
      "Restaurant_longitude": float(rest_lon),
      "Delivery_location_latitude": float(del_lat),
      "Delivery_location_longitude": float(del_lon),
      "distance_km": float(dist_km),
      "order_hour": int(order_hour),
      "order_day_of_week": int(order_dow),
      "GPS_valid": int(gps_valid),
      "distance_valid": int(dist_valid),
      "order_time_missing": int(time_missing),
      "Weather_conditions": str(weather),
      "Road_traffic_density": str(traffic),
      "Type_of_order": str(order_type),
      "Type_of_vehicle": str(veh_type),
      "Festival": str(festival),
      "City": str(city),
  }


# -----------------------------------------------------------------------------
# Load Artifacts & Sidebar Controls
# -----------------------------------------------------------------------------
meta = load_metadata()
model = None
model_error = None

with st.sidebar:
  st.header("⚙️ Model & Lab Controls")
  if MODEL_FILE.exists():
    try:
      model = load_local_model(str(MODEL_FILE))
      st.success(f"Loaded `{MODEL_FILE.name}`")
    except Exception as exc:
      model_error = str(exc)
      st.error(f"Failed loading `{MODEL_FILE.name}`: {exc}")
  else:
    st.warning("`zomato_fast_eta_model.joblib` not found in current folder.")
    uploaded_model = st.file_uploader(
        "Upload .joblib Model File", type=["joblib", "pkl"]
    )
    if uploaded_model is not None:
      try:
        model = joblib.load(uploaded_model)
        st.success("Uploaded model loaded in memory!")
      except Exception as exc:
        st.error(f"Could not load uploaded file: {exc}")

  st.markdown("### 📊 Test Set Benchmark")
  tm = meta.get("test_metrics", DEFAULT_METADATA["test_metrics"])
  c_m1, c_m2 = st.columns(2)
  c_m1.metric("Test R²", f"{tm['r2']:.3f}")
  c_m2.metric("Test MAE", f"{tm['mae_minutes']:.2f}m")
  c_m3, c_m4 = st.columns(2)
  c_m3.metric("Test RMSE", f"{tm['rmse_minutes']:.2f}m")
  c_m4.metric("Within ±5m", f"{tm['within_5_minutes_percent']:.1f}%")

  st.caption(
      f"Model: {meta.get('model_type', 'Random Forest Regressor')} • "
      f"Distance: {meta.get('distance_method', 'Haversine')}"
  )

categories = extract_pipeline_categories(model)

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-card">
    <h1>🛵 Zomato Fast ETA Predictor</h1>
    <p>AI Applications Lab Project • Random Forest Regressor (150 Trees) • Real-Time Haversine Geodesic Distance</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_pred, tab_sim, tab_batch, tab_info = st.tabs([
    "📍 Live ETA Predictor",
    "📈 What-If Traffic & Weather Simulator",
    "📂 Batch CSV Predictor",
    "🧠 Model Card & Feature Importance",
])

# =============================================================================
# TAB 1: LIVE ETA PREDICTOR
# =============================================================================
with tab_pred:
  col_left, col_mid, col_right = st.columns([1.1, 1.15, 1.15], gap="large")

  with col_left:
    st.subheader("1. Rider & Order Profile")
    age = st.slider("Delivery Partner Age", min_value=18, max_value=60, value=27)
    ratings = st.slider(
        "Delivery Partner Rating (★)",
        min_value=1.0,
        max_value=5.0,
        value=4.7,
        step=0.1,
    )
    veh_cond = st.select_slider(
        "Vehicle Condition (0 = Poor, 3 = Excellent)",
        options=[0, 1, 2, 3],
        value=2,
    )
    multi_del = st.selectbox(
        "Active Stacked / Multiple Deliveries", options=[0, 1, 2, 3], index=0
    )
    veh_type = st.selectbox("Vehicle Type", options=categories["Type_of_vehicle"])
    order_type = st.selectbox("Order Category", options=categories["Type_of_order"])

  with col_mid:
    st.subheader("2. Route & Order Timing")
    preset_choice = st.selectbox(
        "Quick Route Preset", options=list(ROUTE_PRESETS.keys()), index=0
    )
    default_coords = ROUTE_PRESETS[preset_choice] or (
        30.7398,
        76.7827,
        30.7463,
        76.6469,
    )

    r_col1, r_col2 = st.columns(2)
    rest_lat = r_col1.number_input(
        "Restaurant Latitude", value=float(default_coords[0]), format="%.4f"
    )
    rest_lon = r_col2.number_input(
        "Restaurant Longitude", value=float(default_coords[1]), format="%.4f"
    )

    d_col1, d_col2 = st.columns(2)
    del_lat = d_col1.number_input(
        "Delivery Latitude", value=float(default_coords[2]), format="%.4f"
    )
    del_lon = d_col2.number_input(
        "Delivery Longitude", value=float(default_coords[3]), format="%.4f"
    )

    calc_dist = haversine_km(rest_lat, rest_lon, del_lat, del_lon)
    override_dist = st.checkbox("Override Haversine Distance manually", value=False)
    if override_dist:
      dist_km = st.number_input(
          "Custom Distance (km)",
          min_value=0.1,
          max_value=100.0,
          value=round(calc_dist, 2),
          step=0.5,
      )
    else:
      dist_km = calc_dist
      st.info(f"📏 **Auto Haversine Distance:** `{dist_km:.2f} km`")

    t_col1, t_col2 = st.columns(2)
    order_hour = t_col1.slider(
        "Order Hour (24h)", min_value=0, max_value=23, value=20
    )
    dow_label = t_col2.selectbox("Day of Week", options=DAY_NAMES, index=5)
    order_dow = DAY_NAMES.index(dow_label)

  with col_right:
    st.subheader("3. Context & Live Prediction")
    w_col1, w_col2 = st.columns(2)
    weather = w_col1.selectbox(
        "Weather Conditions", options=categories["Weather_conditions"]
    )
    traffic_opts = categories["Road_traffic_density"]
    default_tr_idx = (
        traffic_opts.index("Medium")
        if "Medium" in traffic_opts
        else min(1, len(traffic_opts) - 1)
    )
    traffic = w_col2.selectbox(
        "Traffic Density", options=traffic_opts, index=default_tr_idx
    )

    f_col1, f_col2 = st.columns(2)
    festival = f_col1.selectbox("Festival Today?", options=categories["Festival"])
    city = f_col2.selectbox("City Tier", options=categories["City"])

    with st.expander("Data Quality / Validity Flags (Default: Valid)"):
      fl1, fl2, fl3 = st.columns(3)
      gps_valid = int(fl1.checkbox("GPS Valid", value=True))
      dist_valid = int(fl2.checkbox("Distance Valid", value=True))
      time_missing = int(fl3.checkbox("Order Time Missing", value=False))

    input_row = build_feature_row(
        age,
        ratings,
        veh_cond,
        multi_del,
        rest_lat,
        rest_lon,
        del_lat,
        del_lon,
        dist_km,
        order_hour,
        order_dow,
        gps_valid,
        dist_valid,
        time_missing,
        weather,
        traffic,
        order_type,
        veh_type,
        festival,
        city,
    )
    df_single = pd.DataFrame([input_row])

    if model is not None:
      try:
        pred_min = float(np.asarray(model.predict(df_single)).ravel()[0])
        mae = meta.get("test_metrics", {}).get("mae_minutes", 3.214)
        low_eta = max(5.0, pred_min - mae)
        high_eta = pred_min + mae
        arrival_clock = (
            datetime.now() + timedelta(minutes=pred_min)
        ).strftime("%I:%M %p")

        st.markdown(
            f"""
        <div class="eta-box">
            <div class="eta-sub">PREDICTED DELIVERY TIME</div>
            <div class="eta-value">{pred_min:.1f} mins</div>
            <div class="eta-sub">Confidence Window: <b>{low_eta:.0f} – {high_eta:.0f} mins</b> (±{mae:.1f}m MAE)</div>
            <div class="eta-sub" style="margin-top:6px; color:#38bdf8;">Est. Arrival if ordered now: <b>{arrival_clock}</b></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
      except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
      st.warning(
          "Place `zomato_fast_eta_model.joblib` in this directory (or upload it"
          " in the sidebar) to see live Random Forest predictions."
      )

  # Route Map Preview
  st.markdown("#### 🗺️ Geodesic Haversine Route Preview")
  mid_lat = (rest_lat + del_lat) / 2.0
  mid_lon = (rest_lon + del_lon) / 2.0
  points_df = pd.DataFrame([
      {
          "lat": rest_lat,
          "lon": rest_lon,
          "label": "Restaurant (Pickup)",
          "color": [226, 55, 68, 220],
      },
      {
          "lat": del_lat,
          "lon": del_lon,
          "label": "Customer (Dropoff)",
          "color": [34, 197, 94, 220],
      },
  ])
  arc_df = pd.DataFrame([{
      "start_lat": rest_lat,
      "start_lon": rest_lon,
      "end_lat": del_lat,
      "end_lon": del_lon,
  }])

  deck = pdk.Deck(
      initial_view_state=pdk.ViewState(
          latitude=mid_lat, longitude=mid_lon, zoom=11.5, pitch=35
      ),
      layers=[
          pdk.Layer(
              "ArcLayer",
              data=arc_df,
              get_source_position="[start_lon, start_lat]",
              get_target_position="[end_lon, end_lat]",
              get_source_color=[226, 55, 68, 200],
              get_target_color=[56, 189, 248, 200],
              get_width=5,
          ),
          pdk.Layer(
              "ScatterplotLayer",
              data=points_df,
              get_position="[lon, lat]",
              get_fill_color="color",
              get_radius=250,
              pickable=True,
          ),
      ],
      tooltip={"text": "{label}"},
  )
  st.pydeck_chart(deck, use_container_width=True)

# =============================================================================
# TAB 2: WHAT-IF SENSITIVITY SIMULATOR
# =============================================================================
with tab_sim:
  st.subheader("📈 What-If Scenario Analysis (Live Model Sensitivity)")
  st.caption(
      "See how your trained Random Forest Regressor adjusts ETA for the current"
      " order when Traffic, Weather, or Order Hour changes."
  )

  if model is None:
    st.info("Load `zomato_fast_eta_model.joblib` to run live sensitivity sweeps.")
  else:
    sim_col1, sim_col2 = st.columns(2, gap="large")

    with sim_col1:
      st.markdown("##### 🚦 ETA by Road Traffic Density")
      traffic_rows = []
      for tr in categories["Road_traffic_density"]:
        r = dict(input_row)
        r["Road_traffic_density"] = tr
        traffic_rows.append(r)
      df_tr = pd.DataFrame(traffic_rows)
      df_tr["Predicted ETA (min)"] = np.round(model.predict(df_tr), 2)
      st.bar_chart(
          df_tr.set_index("Road_traffic_density")["Predicted ETA (min)"],
          color="#e23744",
      )

      st.markdown("##### 🌦️ ETA by Weather Conditions")
      weather_rows = []
      for w in categories["Weather_conditions"]:
        r = dict(input_row)
        r["Weather_conditions"] = w
        weather_rows.append(r)
      df_w = pd.DataFrame(weather_rows)
      df_w["Predicted ETA (min)"] = np.round(model.predict(df_w), 2)
      st.bar_chart(
          df_w.set_index("Weather_conditions")["Predicted ETA (min)"],
          color="#38bdf8",
      )

    with sim_col2:
      st.markdown("##### 🕒 24-Hour Order Time Curve (00:00 – 23:00)")
      hour_rows = []
      for hr in range(24):
        r = dict(input_row)
        r["order_hour"] = hr
        hour_rows.append(r)
      df_hr = pd.DataFrame(hour_rows)
      df_hr["Predicted ETA (min)"] = np.round(model.predict(df_hr), 2)
      st.line_chart(
          df_hr.set_index("order_hour")["Predicted ETA (min)"], color="#fb7185"
      )

      st.markdown("##### 📏 Distance Sweep (1 km – 25 km)")
      dist_rows = []
      for d_val in range(1, 26, 2):
        r = dict(input_row)
        r["distance_km"] = float(d_val)
        dist_rows.append(r)
      df_d = pd.DataFrame(dist_rows)
      df_d["Predicted ETA (min)"] = np.round(model.predict(df_d), 2)
      st.line_chart(
          df_d.set_index("distance_km")["Predicted ETA (min)"], color="#4ade80"
      )

# =============================================================================
# TAB 3: BATCH CSV PREDICTOR
# =============================================================================
with tab_batch:
  st.subheader("📂 Batch Order ETA Prediction")
  st.write(
      "Upload a CSV containing the 20 model features (or let the app compute"
      " `distance_km` from coordinates) to run bulk predictions."
  )

  b_col1, b_col2 = st.columns([1, 2])
  with b_col1:
    sample_df = pd.DataFrame([
        build_feature_row(
            25,
            4.8,
            2,
            0,
            30.7398,
            76.7827,
            30.7463,
            76.6469,
            13.01,
            19,
            4,
            1,
            1,
            0,
            categories["Weather_conditions"][0],
            categories["Road_traffic_density"][0],
            categories["Type_of_order"][0],
            categories["Type_of_vehicle"][0],
            categories["Festival"][0],
            categories["City"][0],
        ),
        build_feature_row(
            34,
            4.3,
            1,
            1,
            28.6315,
            77.2167,
            28.5494,
            77.2001,
            9.27,
            21,
            5,
            1,
            1,
            0,
            categories["Weather_conditions"][min(3, len(categories["Weather_conditions"]) - 1)],
            categories["Road_traffic_density"][min(3, len(categories["Road_traffic_density"]) - 1)],
            categories["Type_of_order"][min(1, len(categories["Type_of_order"]) - 1)],
            categories["Type_of_vehicle"][min(1, len(categories["Type_of_vehicle"]) - 1)],
            categories["Festival"][0],
            categories["City"][0],
        ),
        build_feature_row(
            29,
            4.9,
            3,
            0,
            12.9784,
            77.6408,
            12.9352,
            77.6245,
            5.12,
            13,
            2,
            1,
            1,
            0,
            categories["Weather_conditions"][min(1, len(categories["Weather_conditions"]) - 1)],
            categories["Road_traffic_density"][min(1, len(categories["Road_traffic_density"]) - 1)],
            categories["Type_of_order"][0],
            categories["Type_of_vehicle"][0],
            categories["Festival"][0],
            categories["City"][0],
        ),
    ])
    st.download_button(
        "⬇️ Download Sample Input CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_zomato_orders.csv",
        mime="text/csv",
        use_container_width=True,
    )

  uploaded_csv = st.file_uploader("Upload Orders CSV", type=["csv"])
  active_batch_df = (
      pd.read_csv(uploaded_csv) if uploaded_csv is not None else sample_df.copy()
  )

  if "distance_km" not in active_batch_df.columns and {
      "Restaurant_latitude",
      "Restaurant_longitude",
      "Delivery_location_latitude",
      "Delivery_location_longitude",
  }.issubset(active_batch_df.columns):
    active_batch_df["distance_km"] = active_batch_df.apply(
        lambda r: haversine_km(
            r["Restaurant_latitude"],
            r["Restaurant_longitude"],
            r["Delivery_location_latitude"],
            r["Delivery_location_longitude"],
        ),
        axis=1,
    )

  if model is not None:
    try:
      req_cols = meta.get("features", DEFAULT_METADATA["features"])
      preds = model.predict(active_batch_df[req_cols])
      active_batch_df["Predicted_Time_taken_min"] = np.round(preds, 2)
      st.dataframe(active_batch_df, use_container_width=True)
      st.download_button(
          "⬇️ Download Predictions CSV",
          data=active_batch_df.to_csv(index=False).encode("utf-8"),
          file_name="zomato_eta_predictions.csv",
          mime="text/csv",
          type="primary",
      )
    except Exception as exc:
      st.error(f"Batch inference error: {exc}")
  else:
    st.dataframe(active_batch_df, use_container_width=True)

# =============================================================================
# TAB 4: MODEL CARD, METRICS & FEATURE IMPORTANCE
# =============================================================================
with tab_info:
  st.subheader("🧠 Model Architecture, Validation & Engineering Notes")

  m_col1, m_col2 = st.columns(2, gap="large")
  with m_col1:
    st.markdown("##### 📊 Validation vs. Test Split Metrics")
    vm = meta.get("validation_metrics", DEFAULT_METADATA["validation_metrics"])
    tm = meta.get("test_metrics", DEFAULT_METADATA["test_metrics"])
    metrics_df = pd.DataFrame({
        "Metric": [
            "RMSE (minutes)",
            "MAE (minutes)",
            "R² Score",
            "Within ±5 Minutes (%)",
        ],
        "Validation Split": [
            vm["rmse_minutes"],
            vm["mae_minutes"],
            vm["r2"],
            vm["within_5_minutes_percent"],
        ],
        "Chronological Test Split": [
            tm["rmse_minutes"],
            tm["mae_minutes"],
            tm["r2"],
            tm["within_5_minutes_percent"],
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("##### ⚙️ Random Forest Hyperparameters")
    hp = meta.get("hyperparameters", DEFAULT_METADATA["hyperparameters"])
    st.json(hp)

    st.markdown("##### 📝 Design & Data Leakage Prevention Notes")
    for note in meta.get("notes", DEFAULT_METADATA["notes"]):
      st.markdown(f"- {note}")

  with m_col2:
    st.markdown("##### 🏆 Top Feature Importances (Extracted from `.joblib`)")
    imp_df = extract_feature_importances(model)
    if imp_df is not None:
      st.bar_chart(
          imp_df.set_index("Feature")["Importance"],
          color="#e23744",
          horizontal=True,
      )
      st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
      st.info(
          "Load `zomato_fast_eta_model.joblib` to inspect live tree feature"
          " importances."
      )

    st.markdown("##### 🧩 Input Feature Schema (20 Features)")
    st.code(
        "\n".join(meta.get("features", DEFAULT_METADATA["features"])),
        language="text",
    )
