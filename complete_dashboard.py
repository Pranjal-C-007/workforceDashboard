import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- ML IMPORTS ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- TRY IMPORTING ADVANCED MODELS ---
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# --- TRY IMPORTING TENSORFLOW/LSTM ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ultimate Workforce Dashboard", layout="wide")

st.title("🚀 Ultimate Workforce Intelligence Dashboard")
st.caption("7 ML Models: Random Forest | XGBoost | LightGBM | CatBoost | Gradient Boosting | AdaBoost | LSTM")

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
            uploaded_file.seek(0)
            first_line = uploaded_file.readline()
            uploaded_file.seek(0)
            
            if not first_line:
                st.sidebar.error("❌ File is empty")
                st.stop()
            
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.sidebar.error("❌ No data rows")
                st.stop()
            
            required_cols = ["user_id", "timestamp", "event_type"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.sidebar.error(f"❌ Missing: {', '.join(missing_cols)}")
                st.stop()
            
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
            st.stop()
    else:
        st.sidebar.info("Upload CSV or use synthetic data")
        data_source = "Generate Synthetic Data"

# --- GENERATE SYNTHETIC DATA ---
if data_source == "Generate Synthetic Data" or df is None:
    dates = [datetime.now() - timedelta(days=x) for x in range(90, 0, -1)]  # 90 days for better LSTM training
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
            # Add some seasonality (higher logins mid-week)
            day_factor = 1 + 0.3 * (1 - abs(date.weekday() - 3) / 3)
            base_logins = random.randint(8, 25)
            
            rows.append({
                "User": user,
                "Department": dept_map.get(user, "Other"),
                "Date": date,
                "Logins": int(base_logins * day_factor),
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

# Build model list based on what's available
model_options = ["Random Forest", "Gradient Boosting", "AdaBoost", "Linear Regression"]

if XGB_AVAILABLE:
    model_options.append("XGBoost")
if LGBM_AVAILABLE:
    model_options.append("LightGBM")
if CATBOOST_AVAILABLE:
    model_options.append("CatBoost")
if TF_AVAILABLE:
    model_options.append("LSTM (Deep Learning)")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    model_options,
    help="Select which ML model to use"
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
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
        except Exception as e:
            st.warning(f"⚠️ {name} failed to train: {e}")
    
    return trained_models

# --- LSTM MODEL (Deep Learning) ---
def train_lstm(data):
    """Train an LSTM model for time series prediction"""
    try:
        # Prepare sequence data
        X = data[features].values
        y = data[target].values
        
        # Reshape for LSTM [samples, timesteps, features]
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with early stopping
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_lstm, y, epochs=50, batch_size=16, verbose=0, callbacks=[early_stop])
        
        return model
    except Exception as e:
        st.warning(f"⚠️ LSTM failed to train: {e}")
        return None

# --- TRAIN ALL MODELS ---
if len(ml_df) > 0:
    trained_models = train_models(ml_df)
    
    # Train LSTM if available and selected
    lstm_model = None
    if TF_AVAILABLE and "LSTM (Deep Learning)" in model_options:
        lstm_model = train_lstm(ml_df)
        if lstm_model is not None:
            trained_models["LSTM (Deep Learning)"] = lstm_model
    
    # Get current model
    current_model = trained_models.get(selected_model)
    
    if current_model is not None:
        # Calculate metrics for all models
        all_metrics = []
        for name, model in trained_models.items():
            try:
                if name == "LSTM (Deep Learning)":
                    # LSTM needs special prediction
                    X_lstm = ml_df[features].values.reshape((ml_df[features].values.shape[0], 1, ml_df[features].values.shape[1]))
                    y_pred = model.predict(X_lstm, verbose=0).flatten()
                else:
                    y_pred = model.predict(ml_df[features])
                
                r2 = r2_score(ml_df[target], y_pred)
                mae = mean_absolute_error(ml_df[target], y_pred)
                all_metrics.append({
                    "Model": name,
                    "R² Score": round(r2 * 100, 1),
                    "MAE": round(mae, 1)
                })
            except Exception as e:
                all_metrics.append({
                    "Model": name,
                    "R² Score": "Error",
                    "MAE": "Error"
                })
        
        metrics_df = pd.DataFrame(all_metrics).sort_values("R² Score", ascending=False)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ML Models",
    "🔥 Burnout Indicators",
    "👤 User Insights",
    "🚨 Anomaly Detection",
    "🔮 Forecast",
    "📋 Executive Summary"
])

# --- TAB 1: ML MODELS ---
with tab1:
    st.subheader("🏆 Model Performance Comparison")
    
    if len(ml_df) > 0 and current_model is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        with col2:
            # Only show models with numeric scores
            plot_df = metrics_df[metrics_df["R² Score"] != "Error"].copy()
            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="R² Score",
                    title="Accuracy Comparison (Higher is Better)",
                    color="R² Score",
                    color_continuous_scale="Blues",
                    text_auto=".1f"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        selected_metrics = metrics_df[metrics_df["Model"] == selected_model]
        if not selected_metrics.empty and selected_metrics.iloc[0]["R² Score"] != "Error":
            st.info(f"**Currently Selected Model: {selected_model}** | R²: {selected_metrics.iloc[0]['R² Score']}% | MAE: {selected_metrics.iloc[0]['MAE']} logins")
        else:
            st.info(f"**Currently Selected Model: {selected_model}**")
        
        # --- FEATURE IMPORTANCE (Only for tree-based models) ---
        st.subheader("🔍 Feature Importance (What Drives Logins?)")
        
        tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]
        
        if selected_model in tree_models and selected_model in trained_models:
            model = trained_models[selected_model]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
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
        elif selected_model == "LSTM (Deep Learning)":
            st.info("ℹ️ LSTM models don't provide traditional feature importance. They learn complex temporal patterns.")
        else:
            st.info("ℹ️ This model doesn't provide feature importance or isn't available.")
        
        # --- ACTUAL VS PREDICTED ---
        st.subheader(f"📈 Actual vs Predicted Logins ({selected_model})")
        
        try:
            if selected_model == "LSTM (Deep Learning)" and lstm_model is not None:
                X_lstm = ml_df[features].values.reshape((ml_df[features].values.shape[0], 1, ml_df[features].values.shape[1]))
                y_pred_selected = lstm_model.predict(X_lstm, verbose=0).flatten()
            else:
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
        except Exception as e:
            st.warning(f"⚠️ Could not generate predictions: {e}")
    else:
        st.warning("⚠️ Not enough data for ML training.")

# --- TAB 2: BURNOUT ---
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
        st.error(f"⚠️ High burnout risk: {', '.join(high_risk['User'].tolist())}")

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
        
        st.subheader("📈 Activity Trend")
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
    
    if len(ml_df) > 0 and current_model is not None:
        try:
            future_dates = [datetime.now() + timedelta(days=x) for x in range(1, 8)]
            future_features = pd.DataFrame({
                "DayOfWeek": [d.weekday() for d in future_dates],
                "IsWeekend": [1 if d.weekday() >= 5 else 0 for d in future_dates],
                "LockEvents": [random.randint(5, 20) for _ in range(7)],
                "ActiveHours": [round(random.uniform(6, 9), 1) for _ in range(7)]
            })
            
            if selected_model == "LSTM (Deep Learning)" and lstm_model is not None:
                X_lstm = future_features.values.reshape((future_features.values.shape[0], 1, future_features.values.shape[1]))
                future_predictions = lstm_model.predict(X_lstm, verbose=0).flatten()
            else:
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
            
            if not metrics_df.empty and metrics_df[metrics_df["Model"] == selected_model].iloc[0]["R² Score"] != "Error":
                model_metrics = metrics_df[metrics_df["Model"] == selected_model].iloc[0]
                st.info(f"**Model Confidence:** R² Score: {model_metrics['R² Score']}%")
        except Exception as e:
            st.warning(f"⚠️ Could not generate forecast: {e}")
    else:
        st.warning("⚠️ Not enough data for forecast.")

# --- TAB 6: EXECUTIVE SUMMARY ---
with tab6:
    st.subheader("📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    total_logins = filtered_df["Logins"].sum()
    avg_active_hours = filtered_df["ActiveHours"].mean()
    total_anomalies = len(anomalies) if len(daily) > 0 else 0
    total_users = filtered_df["User"].nunique()
    
    col1.metric("Total Logins", f"{total_logins:,}")
    col2.metric("Avg Active Hours", f"{avg_active_hours:.1f}")
    col3.metric("Anomalies", total_anomalies)
    col4.metric("Active Users", total_users)
    
    # Model performance summary
    st.subheader("🏆 Best Performing Model")
    if not metrics_df.empty:
        best_model = metrics_df.iloc[0]
        if best_model["R² Score"] != "Error":
            st.success(f"✅ **{best_model['Model']}** | R²: {best_model['R² Score']}% | MAE: {best_model['MAE']} logins")
        else:
            st.info("Model metrics not available")
    
    # Burnout summary
    st.subheader("🔥 Burnout Risk Distribution")
    if "burnout_df" in locals() and not burnout_df.empty:
        burnout_summary = burnout_df.groupby("Risk").size().reset_index(name="Count")
        if not burnout_summary.empty:
            fig = px.pie(
                burnout_summary,
                values="Count",
                names="Risk",
                title="Burnout Risk Distribution",
                color="Risk",
                color_discrete_map={
                    "🔴 High": "#ff6b6b",
                    "🟡 Medium": "#ffd93d",
                    "🟢 Low": "#6bcb77"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    insights = []
    
    if len(user_stats) > 0:
        top_user = user_stats.sort_values("Logins", ascending=False).iloc[0]
        insights.append(f"🏆 **Most Active User:** {top_user['User']} with {top_user['Logins']} logins")
    
    if not burnout_df.empty and not burnout_df[burnout_df["Risk"] == "🔴 High"].empty:
        insights.append(f"⚠️ **Burnout Alert:** {len(burnout_df[burnout_df['Risk'] == '🔴 High'])} users at high risk")
    else:
        insights.append("✅ **All users:** Burnout risk is low")
    
    if total_anomalies > 0:
        insights.append(f"🚨 **Anomaly Alert:** {total_anomalies} unusual activity days detected")
    else:
        insights.append("✅ **No anomalies:** All activity is within normal range")
    
    if not metrics_df.empty and metrics_df.iloc[0]["R² Score"] != "Error":
        best_model = metrics_df.iloc[0]["Model"]
        best_score = metrics_df.iloc[0]["R² Score"]
        insights.append(f"🧠 **Best ML Model:** {best_model} (R²: {best_score}%)")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    recommendations = []
    
    if not burnout_df.empty and not burnout_df[burnout_df["Risk"] == "🔴 High"].empty:
        recommendations.append("🔴 **Immediate:** Schedule 1-on-1 meetings with high-risk employees")
    else:
        recommendations.append("🟢 **Good:** No immediate burnout concerns")
    
    if total_anomalies > 0:
        recommendations.append("📊 **Review:** Investigate anomaly days for potential causes")
    
    if not recommendations:
        recommendations.append("✅ **All systems normal.** Continue monitoring")
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

st.divider()
st.caption("🔹 Ultimate Dashboard | 7 ML Models: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, LSTM")