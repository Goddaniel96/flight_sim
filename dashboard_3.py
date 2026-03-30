import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from pathlib import Path

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(layout="wide", page_title="Fuel Optimization Analysis")
st.title("✈️ Fuel Optimization & Analysis")

# --- 2. ฟังก์ชันโหลดโมเดล (รองรับ Deployment) ---
# หา Path ของโฟลเดอร์ที่สคริปต์นี้ตั้งอยู่
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
model_path = current_dir / 'fuel_model.pkl'
columns_path = current_dir / 'model_columns.pkl'

@st.cache_resource
def load_assets():
    # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงไหม
    if not model_path.exists() or not columns_path.exists():
        st.error(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        st.info("ไฟล์ที่ระบบมองเห็นในตอนนี้: " + str(os.listdir(current_dir)))
        return None, None
    
    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด: {e}")
        return None, None

model, model_columns = load_assets()

# --- 3. Sidebar Inputs ---
st.sidebar.header("🕹️ Adjust Parameters")
s_zfw = st.sidebar.slider("Zero Fuel Weight (kg)", 45000, 180000, 60000, 1000)
s_duration = st.sidebar.slider("Duration (min)", 60, 720, 180, 15)
s_altitude = st.sidebar.slider("Altitude (ft)", 30000, 41000, 36000, 1000)
s_headwind = st.sidebar.slider("Wind (kts) [+Head / -Tail]", -50, 50, 0, 1)
s_engine = st.sidebar.slider("Aircraft Performance Factor", 0.90, 1.10, 1.0, 0.01)
s_ac_type = st.sidebar.selectbox("Aircraft Type", ['A320', 'A330', 'B737', 'B777', 'A350', 'B787'])

# --- 4. ฟังก์ชันเตรียมข้อมูล ---
def build_feature_df(params, col_list):
    data = {
        'zero_fuel_weight_kg': params['zfw'],
        'planned_duration_min': params['duration'],
        'planned_altitude_ft': params['altitude'],
        'avg_headwind_kts': params['wind'],
        'engine_health_modifier': params['engine'],
        'aircraft_type_A330': 1 if params['ac_type'] == 'A330' else 0,
        'aircraft_type_B737': 1 if params['ac_type'] == 'B737' else 0,
        'aircraft_type_B777': 1 if params['ac_type'] == 'B777' else 0,
        'aircraft_type_A350': 1 if params['ac_type'] == 'A350' else 0,
        'aircraft_type_B787': 1 if params['ac_type'] == 'B787' else 0,
    }
    df = pd.DataFrame([data])
    if col_list:
        df = df.reindex(columns=col_list, fill_value=0)
    return df

# --- 5. Main Interface ---
st.subheader("1. Input Data Selection")
input_mode = st.radio("Choose Input Method:", ["Manual (Sidebar Sliders)", "Upload File (CSV/Excel)"], horizontal=True)

final_params = {}
if input_mode == "Upload File (CSV/Excel)":
    uploaded_file = st.file_uploader("Attach Flight Data", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df_file = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            row = df_file.iloc[0]
            final_params = {
                'zfw': row.get('zero_fuel_weight_kg', s_zfw),
                'duration': row.get('planned_duration_min', s_duration),
                'altitude': row.get('planned_altitude_ft', s_altitude),
                'wind': row.get('avg_headwind_kts', s_headwind),
                'engine': row.get('engine_health_modifier', s_engine),
                'ac_type': row.get('aircraft_type', s_ac_type)
            }
            st.success("Using data from uploaded file")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            input_mode = "Manual (Sidebar Sliders)"
    else:
        input_mode = "Manual (Sidebar Sliders)"

if input_mode == "Manual (Sidebar Sliders)":
    final_params = {'zfw': s_zfw, 'duration': s_duration, 'altitude': s_altitude, 'wind': s_headwind, 'engine': s_engine, 'ac_type': s_ac_type}

# --- 6. Prediction ---
if model is not None and model_columns is not None:
    input_df = build_feature_df(final_params, model_columns)
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("2. Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Fuel Burn", f"{prediction:,.2f} kg")
    c2.metric("Selected Aircraft", final_params['ac_type'])
    c3.metric("Input Source", "File" if input_mode != "Manual (Sidebar Sliders)" else "Manual")

    with st.expander("🔍 Debug: Raw Data to Model"):
        st.write(input_df)

    # --- 7. Visualization ---
    st.markdown("---")
    st.subheader("3. Risk Index")
    risk_val = 0
    if final_params['wind'] > 25: risk_val += 5
    if final_params['engine'] > 1.04: risk_val += 3
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_val,
        gauge = {'axis': {'range': [0, 10]},
                 'steps': [{'range': [0, 4], 'color': "green"}, {'range': [4, 7], 'color': "orange"}, {'range': [7, 10], 'color': "red"}]}
    ))
    # แก้ไข use_container_width เป็น width='stretch' ตามเวอร์ชันใหม่
    st.plotly_chart(fig, width='stretch')
else:
    st.warning("⚠️ Dashboard is ready, but model is not loaded. Please check the error message above.")
