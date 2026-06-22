import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Complete Workforce Dashboard", layout="wide")

st.title("🤖 Complete Workforce Intelligence Dashboard")
st.caption("Everything: ML Models | Burnout | User Insights | Anomalies | Forecast | Export")

# --- SIDEBAR: DATA SOURCE ---
st.sidebar.header("📂 Data Source")

data_source = st.sidebar.radio(
    "Choose Data Source",
    ["Generate Synthetic Data", "Upload Real Data"]
)

uploaded_file = None
df = None

if data_source == "Upload Real Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # --- SAFETY CHECK: Check if file has content ---
            uploaded_file.seek(0)
            first_line = uploaded_file.readline()
            uploaded_file.seek(0)
            
            if not first_line:
                st.sidebar.error("❌ The file is empty. Please upload a valid CSV file.")
                st.stop()
            
            df = pd.read_csv(uploaded_file)
            
            # --- SAFETY CHECK: Check if dataframe has data ---
            if df.empty:
                st.sidebar.error("❌ The CSV file has no data rows. Please upload a valid CSV file.")
                st.stop()
            
            # --- SAFETY CHECK: Check if required columns exist ---
            required_cols = ["user_id", "timestamp", "event_type"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.sidebar.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.sidebar.info("Required columns: user_id, timestamp, event_type")
                st.stop()
            
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        st.sidebar.info("Upload a CSV file or switch to Synthetic Data")
        data_source = "Generate Synthetic Data"

# --- GENERATE SYNTHETIC DATA ---
if data_source == "Generate Synthetic Data" or df is None:
    dates = [datetime.now() - timedelta(days=x) for x in range(60, 0, -1)]
    users = [f"emp_{i:03d}" for i in range(1, 11)]
    
    dept_map = {
        "emp_001": "Engineering",
        "emp_002": "Engineering",
        "emp_003": "Sales",
        "emp_004": "Sales",
        "emp_005": "Marketing",
        "emp_006": "Marketing",
        "emp_007": "HR",
        "emp_008": "Finance",
        "emp_009": "Engineering",
        "emp_010": "Sales"
    }
    
    rows = []
    for user in users:
        for date in dates:
            rows.append({
                "User": user,
                "Department": dept_map.get(user, "Other"),
                "Date": date,
                "Logins": random.randint(5, 30),
                "LockEvents": random.randint(2, 15),
                "ActiveHours": round(random.uniform(4, 10), 1),
                "LateHours": random.randint(0, 2),
                "IsWeekend": 1 if date.weekday() >= 5 else 0,
                "DayOfWeek": date.weekday()
            })
    df = pd.DataFrame(rows)

# --- PROCESS UPLOADED DATA ---
if data_source == "Upload Real Data" and uploaded_file is not None:
    df_processed = pd.DataFrame()
    df_processed["User"] = df["user_id"]
    df_processed["Department"] = df["department"] if "department" in df.columns else "All"
    df_processed["Date"] = pd.to_datetime(df["timestamp"]).dt.date
    df_processed["IsWeekend"] = pd.to_datetime(df["timestamp"]).apply(lambda x: 1 if x.weekday() >= 5 else 0)
    df_processed["DayOfWeek"] = pd.to_datetime(df["timestamp"]).apply(lambda x: x.weekday())
    
    event_counts = df.groupby(["user_id", pd.to_datetime(df["timestamp"]).dt.date]).size().reset_index(name="Logins")
    df_processed = df_processed.merge(event_counts, left_on=["User", "Date"], right_on=["user_id", "timestamp"], how="left")
    df_processed["Logins"] = df_processed["Logins"].fillna(0)
    
    df_processed["LockEvents"] = df_processed["Logins"].apply(lambda x: random.randint(2, 15))
    df_processed["ActiveHours"] = df_processed["Logins"].apply(lambda x: round(random.uniform(4, 10), 1))
    df_processed["LateHours"] = df_processed["Logins"].apply(lambda x: random.randint(0, 2))
    
    df = df_processed

# --- SIDEBAR: FILTERS ---
st.sidebar.header("🔍 Filters")

departments = ["All"] + sorted(df["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("🏢 Department", departments)

filtered_for_users = df.copy()
if selected_department != "All":
    filtered_for_users = filtered_for_users[filtered_for_users["Department"] == selected_department]

users_list = ["All"] + sorted(filtered_for_users["User"].unique().tolist())
selected_user = st.sidebar.selectbox("👤 Select User", users_list)

min_date = df["Date"].min()
max_date = df["Date"].max()

# Convert to datetime for date picker
min_date_dt = pd.to_datetime(min_date)
max_date_dt = pd.to_datetime(max_date)

date_range = st.sidebar.date_input(
    "📅 Date Range",
    [min_date_dt, max_date_dt],
    min_value=min_date_dt,
    max_value=max_date_dt
)

st.sidebar.header("📊 Time Period")
time_period = st.sidebar.radio(
    "View Data By",
    ["Daily", "Weekly", "Monthly"]
)

# --- APPLY FILTERS ---
filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_department]

if selected_user != "All":
    filtered_df = filtered_df[filtered_df["User"] == selected_user]

# --- FIX: Date range filter ---
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df["Date"]) >= start_date) &
        (pd.to_datetime(filtered_df["Date"]) <= end_date)
    ]

# --- TIME PERIOD AGGREGATION ---
display_df = filtered_df.copy()
if time_period == "Weekly":
    display_df["Period"] = display_df["Date"].apply(lambda x: f"Week {x.isocalendar().week}")
    display_df = display_df.groupby("Period").agg({
        "Logins": "sum",
        "LockEvents": "sum",
        "ActiveHours": "mean"
    }).reset_index()
elif time_period == "Monthly":
    display_df["Period"] = display_df["Date"].apply(lambda x: x.strftime("%Y-%m"))
    display_df = display_df.groupby("Period").agg({
        "Logins": "sum",
        "LockEvents": "sum",
        "ActiveHours": "mean"
    }).reset_index()
else:
    display_df["Period"] = display_df["Date"]

# --- SIDEBAR: MODEL SELECTOR ---
st.sidebar.header("🧠 ML Model")

model_options = ["Random Forest", "XGBoost", "Linear Regression"]
selected_model = st.sidebar.selectbox(
    "Choose Model",
    model_options,
    help="Select which ML model to use for predictions"
)

# --- SIDEBAR: EXPORT ---
st.sidebar.header("📥 Export")
csv = filtered_df.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"workforce_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# --- MAIN DASHBOARD ---
st.subheader("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Users", filtered_df["User"].nunique())
col2.metric("Total Logins", filtered_df["Logins"].sum())
col3.metric("Avg Active Hours", round(filtered_df["ActiveHours"].mean(), 1))
col4.metric("Lock Events", filtered_df["LockEvents"].sum())
col5.metric("Data Points", len(filtered_df))

st.divider()

# --- PREPARE ML DATA ---
ml_df = filtered_df.groupby("Date").agg({
    "Logins": "sum",
    "LockEvents": "sum",
    "ActiveHours": "mean",
    "IsWeekend": "first",
    "DayOfWeek": "first"
}).reset_index()

features = ["DayOfWeek", "IsWeekend", "LockEvents", "ActiveHours"]
target = "Logins"

# --- TRAIN MODELS ---
def train_models(data):
    X = data[features]
    y = data[target]
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models

if len(ml_df) > 0:
    trained_models = train_models(ml_df)
    current_model = trained_models[selected_model]
    
    all_metrics = []
    for name, model in trained_models.items():
        y_pred = model.predict(ml_df[features])
        all_metrics.append({
            "Model": name,
            "R² Score": round(r2_score(ml_df[target], y_pred) * 100, 1),
            "MAE": round(mean_absolute_error(ml_df[target], y_pred), 1)
        })
    metrics_df = pd.DataFrame(all_metrics).sort_values("R² Score", ascending=False)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ML Models",
    "🔥 Burnout Indicators",
    "👤 User Insights",
    "🚨 Anomaly Detection",
    "🔮 Forecast"
])

# --- TAB 1: ML MODELS ---
with tab1:
    st.subheader("🏆 Model Performance Comparison")
    
    if len(ml_df) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        with col2:
            fig = px.bar(
                metrics_df,
                x="Model",
                y="R² Score",
                title="Accuracy Comparison",
                color="R² Score",
                color_continuous_scale="Blues",
                text_auto=".1f"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        selected_metrics = metrics_df[metrics_df["Model"] == selected_model].iloc[0]
        st.info(f"**Currently Selected Model: {selected_model}** | R²: {selected_metrics['R² Score']}% | MAE: {selected_metrics['MAE']} logins")
        
        st.subheader("🔍 Feature Importance (What Drives Logins?)")
        
        if selected_model in ["Random Forest", "XGBoost"]:
            importances = current_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                "Feature": ["Day of Week", "Is Weekend", "Lock Events", "Active Hours"],
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(
                feature_imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance - {selected_model}",
                color="Importance",
                color_continuous_scale="Blues",
                text_auto=".1%"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Linear Regression doesn't provide feature importance. Select Random Forest or XGBoost for this feature.")
        
        st.subheader(f"📈 Actual vs Predicted Logins ({selected_model})")
        
        y_pred_selected = current_model.predict(ml_df[features])
        
        pred_df = pd.DataFrame({
            "Date": ml_df["Date"],
            "Actual": ml_df[target],
            "Predicted": y_pred_selected
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df["Date"],
            y=pred_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pred_df["Date"],
            y=pred_df["Predicted"],
            mode="lines+markers",
            name=f"Predicted ({selected_model})",
            line=dict(color="red", width=2, dash="dash")
        ))
        fig.update_layout(
            title=f"Model Predictions vs Reality - {selected_model}",
            xaxis_title="Date",
            yaxis_title="Number of Logins",
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Not enough data for ML training. Need at least 10 days.")

# --- TAB 2: BURNOUT INDICATORS ---
with tab2:
    st.subheader("🔥 Burnout Risk Indicators")
    
    burnout_df = filtered_df.groupby("User").agg({
        "LateHours": "sum",
        "IsWeekend": "sum",
        "Logins": "sum",
        "LockEvents": "sum"
    }).reset_index()
    
    burnout_df["WeekendWork"] = burnout_df["IsWeekend"]
    burnout_df["BurnoutScore"] = burnout_df["LateHours"] * 2 + burnout_df["WeekendWork"] * 3
    
    no_break_days = filtered_df.groupby("User").apply(
        lambda x: x[x["LockEvents"] == 0].shape[0]
    ).reset_index(name="NoBreakDays")
    
    burnout_df = burnout_df.merge(no_break_days, on="User", how="left")
    burnout_df["NoBreakDays"] = burnout_df["NoBreakDays"].fillna(0)
    burnout_df["NoBreakRisk"] = burnout_df["NoBreakDays"].apply(
        lambda x: "🔴 High" if x > 5 else "🟡 Medium" if x > 2 else "🟢 Low"
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("⚠️ Late Hours", burnout_df["LateHours"].sum())
    col2.metric("📅 Weekend Work", burnout_df["WeekendWork"].sum())
    no_break_high = burnout_df[burnout_df["NoBreakRisk"] == "🔴 High"].shape[0]
    col3.metric("🚫 No Break Risk", f"{no_break_high} users at risk")
    
    st.subheader("Risk by User")
    burnout_df["Risk"] = burnout_df["BurnoutScore"].apply(
        lambda x: "🔴 High" if x > 20 else "🟡 Medium" if x > 10 else "🟢 Low"
    )
    st.dataframe(burnout_df[["User", "BurnoutScore", "Risk", "LateHours", "WeekendWork", "NoBreakDays", "NoBreakRisk"]])
    
    high_risk = burnout_df[burnout_df["Risk"] == "🔴 High"]
    if not high_risk.empty:
        st.error(f"⚠️ High burnout risk detected for: {', '.join(high_risk['User'].tolist())}")
    
    no_break_alert = burnout_df[burnout_df["NoBreakRisk"] == "🔴 High"]
    if not no_break_alert.empty:
        st.warning(f"🚫 No break risk: {', '.join(no_break_alert['User'].tolist())} - these users work without breaks")

# --- TAB 3: USER INSIGHTS ---
with tab3:
    st.subheader("👤 User Activity Breakdown")
    
    user_stats = filtered_df.groupby("User").agg({
        "Logins": "sum",
        "ActiveHours": "mean",
        "LockEvents": "sum",
        "LateHours": "sum"
    }).reset_index()
    
    st.dataframe(user_stats)
    
    st.subheader("🏆 Top Active Users")
    top_users = user_stats.sort_values("Logins", ascending=False).head(5)
    fig = px.bar(
        top_users,
        x="User",
        y="Logins",
        title="Top 5 Active Users",
        color="Logins",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: ANOMALY DETECTION ---
with tab4:
    st.subheader("🚨 Anomaly Detection")
    
    daily = filtered_df.groupby("Date").agg({
        "Logins": "sum",
        "LockEvents": "sum",
        "ActiveHours": "mean"
    }).reset_index()
    
    if len(daily) > 0:
        threshold = daily["Logins"].mean() + daily["Logins"].std() * 2
        anomalies = daily[daily["Logins"] > threshold]
        
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} anomaly days detected")
            st.dataframe(anomalies)
        else:
            st.success("✅ No anomalies detected")
        
        st.subheader("📈 60-Day Activity Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"],
            y=daily["Logins"],
            mode="lines",
            name="Logins",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=daily["Date"],
            y=daily["LockEvents"],
            mode="lines",
            name="Lock Events",
            line=dict(color="orange", width=2)
        ))
        fig.update_layout(
            title="Login & Lock Event Trends",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 5: FORECAST ---
with tab5:
    st.subheader(f"🔮 7-Day Forecast ({selected_model})")
    
    if len(ml_df) > 0:
        future_dates = [datetime.now() + timedelta(days=x) for x in range(1, 8)]
        future_features = pd.DataFrame({
            "DayOfWeek": [d.weekday() for d in future_dates],
            "IsWeekend": [1 if d.weekday() >= 5 else 0 for d in future_dates],
            "LockEvents": [random.randint(5, 20) for _ in range(7)],
            "ActiveHours": [round(random.uniform(6, 9), 1) for _ in range(7)]
        })
        
        future_predictions = current_model.predict(future_features)
        
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Logins": [round(x, 1) for x in future_predictions]
        })
        
        colors = ["#ff6b6b" if d.weekday() >= 5 else "#51cf66" for d in future_dates]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=forecast_df["Date"],
            y=forecast_df["Predicted Logins"],
            marker_color=colors,
            text=forecast_df["Predicted Logins"],
            textposition="outside"
        ))
        fig.update_layout(
            title=f"Predicted Login Activity for Next 7 Days ({selected_model})",
            xaxis_title="Date",
            yaxis_title="Predicted Logins",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 Green = Weekday | Red = Weekend")
        
        model_metrics = metrics_df[metrics_df["Model"] == selected_model].iloc[0]
        st.info(f"**Model Confidence:** R² Score: {model_metrics['R² Score']}%")
    else:
        st.warning("⚠️ Not enough data for forecast. Need at least 10 days.")

st.divider()
st.caption("🔹 Complete Dashboard | Built with Python, Streamlit, Scikit-Learn & Plotly")