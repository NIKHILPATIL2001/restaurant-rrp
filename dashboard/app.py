"""
Streamlit dashboard for the Restaurant RRP Forecaster.

Pages:
  - Home: tomorrow's covers, staff grid, ingredient orders
  - Convergence: three overlaid lines (actual / baseline / corrected) + delta panel
  - Corrections: submit a manager correction form

Cold-start banner shown when regime != "full".
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("RRP_API_URL", "http://localhost:8000")
TOMORROW = str(date.today() + timedelta(days=1))


def api(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_URL}{path}"
    try:
        r = httpx.request(method, url, timeout=15.0, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


st.set_page_config(
    page_title="Restaurant RRP Forecaster",
    page_icon="🍽",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Forecast", "Convergence", "Submit Correction"])

# ─── FORECAST PAGE ────────────────────────────────────────────────────────────
if page == "Forecast":
    st.title("Tomorrow's Forecast")
    st.caption(f"Forecasting for: **{TOMORROW}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Hourly Covers")
        covers_data = api("POST", "/v1/forecast/covers", json={"date": TOMORROW, "restaurant_id": 1})
        if covers_data:
            regime = covers_data.get("regime", "unknown")
            if regime != "full":
                st.warning(f"Cold-start mode: regime = **{regime}**. Predictions use prior estimates.")
            if covers_data.get("degraded_features"):
                st.info(f"Degraded features: {', '.join(covers_data['degraded_features'])}")

            import pandas as pd
            hourly = covers_data["hourly_covers"]
            df = pd.DataFrame({"hour": list(range(24)), "covers": hourly})
            st.bar_chart(df.set_index("hour"))
            st.metric("Daily Total", f"{covers_data['daily_total']:.0f} covers")
            st.caption(f"Confidence band: ±{covers_data['confidence_band']*100:.0f}%")
            if not covers_data.get("online_layer_active"):
                st.caption("Online layer: not yet active (no corrections received)")

    with col2:
        st.subheader("Staff Schedule")
        staff_data = api("POST", "/v1/forecast/staff", json={"date": TOMORROW, "restaurant_id": 1})
        if staff_data and staff_data.get("schedule"):
            import pandas as pd
            schedule = staff_data["schedule"]
            df = pd.DataFrame(schedule, index=[f"{h}:00" for h in range(24)])
            st.dataframe(df, width="stretch")

            station_load = staff_data.get("station_load") or []
            if station_load:
                st.caption("Station peak load (cook-minutes/hour)")
                station_df = pd.DataFrame([
                    {"station": s["station"], "peak_hour": s["peak_hour"], "peak_minutes": s["peak_minutes"]}
                    for s in station_load
                ])
                st.dataframe(station_df, width="stretch", hide_index=True)

    with col3:
        st.subheader("Ingredient Orders")
        inv_data = api(
            "POST", "/v1/forecast/inventory",
            json={"date": TOMORROW, "restaurant_id": 1, "horizon_days": 3}
        )
        if inv_data and inv_data.get("orders"):
            for order in inv_data["orders"]:
                risk_label = "🔴 STOCKOUT RISK" if order["stockout_risk"] else ""
                st.write(
                    f"**{order['ingredient_name']}** — {order['order_qty']} {order['unit']} {risk_label}"
                )
                with st.expander("Details"):
                    st.caption(order["rationale"])
                    st.caption(f"On hand: {order['on_hand']} | Lead time: {order['lead_time_days']}d | Shelf life: {order['shelf_life_days']}d")

# ─── CONVERGENCE PAGE ─────────────────────────────────────────────────────────
elif page == "Convergence":
    st.title("Convergence: Feedback Loop Proof")
    st.caption(
        "Shows baseline MAPE (LightGBM only) vs corrected MAPE (LightGBM + online residual). "
        "The gap widening over time is the proof that the feedback loop is doing work."
    )

    surface = st.selectbox("Surface", ["covers", "staff", "inventory"], index=0)
    conv_data = api("GET", f"/v1/metrics/convergence?surface={surface}")

    if conv_data and conv_data.get("series"):
        import pandas as pd

        series = conv_data["series"]
        df = pd.DataFrame(series)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        summary = conv_data.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Days tracked", summary.get("n_days", 0))
        c2.metric(
            "Avg Baseline MAPE",
            f"{(summary.get('avg_baseline_mape') or 0)*100:.1f}%"
            if summary.get("avg_baseline_mape") else "—"
        )
        c3.metric(
            "Avg Corrected MAPE",
            f"{(summary.get('avg_corrected_mape') or 0)*100:.1f}%"
            if summary.get("avg_corrected_mape") else "—",
            delta=f"-{summary.get('mape_improvement_pct', 0):.1f}% improvement"
            if summary.get("mape_improvement_pct") else None,
            delta_color="inverse",
        )

        # Three overlaid lines
        st.subheader("MAPE over time")
        chart_df = df[["date", "baseline_mape", "corrected_mape"]].set_index("date").dropna()
        if not chart_df.empty:
            st.line_chart(chart_df)

        # Delta panel
        st.subheader("Delta (baseline − corrected MAPE)")
        if "baseline_mape" in df.columns and "corrected_mape" in df.columns:
            df["delta"] = df["baseline_mape"] - df["corrected_mape"]
            st.area_chart(df[["date", "delta"]].set_index("date").dropna())
            st.caption(
                "Positive delta = online residual improving over baseline. "
                "A widening delta is the visual proof of feedback-loop convergence."
            )

        st.subheader("Raw data")
        st.dataframe(
            df[["date", "baseline_mape", "corrected_mape", "n_corrections", "model_version"]],
            width="stretch",
        )
    else:
        st.info(
            "No convergence data yet. Run `python -m rrp.scripts.simulate --days 60 --seed 42` "
            "or wait for the background warmup to complete."
        )

# ─── SUBMIT CORRECTION PAGE ───────────────────────────────────────────────────
elif page == "Submit Correction":
    st.title("Submit Manager Correction")
    st.caption("Corrections are immediately fed into the online residual learner.")

    with st.form("correction_form"):
        scope = st.selectbox("Scope", ["covers", "staff", "inventory"])
        corr_date = st.date_input("Date", value=date.today() - timedelta(days=1))
        hour = st.slider("Hour (for covers/staff)", 0, 23, 19)
        predicted = st.number_input("Predicted value", min_value=0.0, value=120.0)
        actual = st.number_input("Actual value", min_value=0.0, value=85.0)
        reason = st.selectbox(
            "Reason code",
            ["unknown", "rain", "event", "holiday", "closure", "staff_shortage", "pos_error"]
        )
        note = st.text_input("Note (optional)")
        scope_id = st.text_input("Role/Ingredient ID (for staff/inventory, leave blank for covers)")
        submitted = st.form_submit_button("Submit Correction")

    if submitted:
        payload: dict[str, Any] = {
            "scope": scope,
            "date": str(corr_date),
            "hour": hour,
            "predicted": predicted,
            "actual": actual,
            "reason_code": reason,
            "note": note or None,
            "scope_id": scope_id or None,
        }
        result = api("POST", "/v1/corrections", json=payload)
        if result:
            if result.get("quarantined"):
                st.warning(f"Correction quarantined: {result['message']}")
            else:
                st.success(f"Correction applied. {result['message']}")
                st.caption("Refresh the Convergence page to see the updated MAPE.")
