import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(layout="wide", page_title="Fuel Optimization Analysis")
st.title("✈️ Fuel Optimization & Analysis")

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_assets(model_path='fuel_model.pkl', columns_path='model_columns.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None
    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns
    except:
        return None, None

model, model_columns = load_assets()

# --- 3. Sidebar Inputs ---
st.sidebar.header("🕹️ Adjust Parameters")

# สร้าง Slider และเก็บค่าไว้ในตัวแปร
s_zfw = st.sidebar.slider("Zero Fuel Weight (kg)", 45000, 180000, 60000, 1000)
s_duration = st.sidebar.slider("Duration (min)", 60, 720, 180, 15)
s_altitude = st.sidebar.slider("Altitude (ft)", 30000, 41000, 36000, 1000)
s_headwind = st.sidebar.slider("Wind (kts) [+Head / -Tail]", -50, 50, 0, 1)
s_engine = st.sidebar.slider("Aircraft Performance Factor", 0.90, 1.10, 1.0, 0.01)
s_ac_type = st.sidebar.selectbox("Aircraft Type", ['A320', 'A330', 'B737', 'B777', 'A350', 'B787'])

# --- 4. Main Interface ---
st.subheader("1. Input Data Selection")
input_mode = st.radio("Choose Input Method:", ["Manual (Sidebar Sliders)", "Upload File (CSV/Excel)"], horizontal=True)

final_params = {}

if input_mode == "Upload File (CSV/Excel)":
    uploaded_file = st.file_uploader("Attach Flight Data", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df_file = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            row = df_file.iloc[0]
            
            # Mapping ค่าจากไฟล์ (ถ้าไม่มีใช้ค่าจาก Slider เป็น Default)
            final_params = {
                'zfw': row.get('zero_fuel_weight_kg', s_zfw),
                'duration': row.get('planned_duration_min', s_duration),
                'altitude': row.get('planned_altitude_ft', s_altitude),
                'wind': row.get('avg_headwind_kts', s_headwind),
                'engine': row.get('engine_health_modifier', s_engine),
                'ac_type': row.get('aircraft_type', s_ac_type)
            }
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Waiting for file... (Using Sidebar values for now)")
        input_mode = "Manual (Sidebar Sliders)"

if input_mode == "Manual (Sidebar Sliders)":
    final_params = {
        'zfw': s_zfw,
        'duration': s_duration,
        'altitude': s_altitude,
        'wind': s_headwind,
        'engine': s_engine,
        'ac_type': s_ac_type
    }

# --- 5. Data Preparation (Logic สำคัญที่ทำให้ค่าขยับ) ---
def build_feature_df(params, col_list):
    # สร้าง dict ให้ชื่อ key ตรงกับตอน Train (ต้องเช็คกับ model_columns.pkl)
    data = {
        'zero_fuel_weight_kg': params['zfw'],
        'planned_duration_min': params['duration'],
        'planned_altitude_ft': params['altitude'],
        'avg_headwind_kts': params['wind'],
        'engine_health_modifier': params['engine'],
        'aircraft_type_A330': 1 if params['ac_type'] == 'A330' else 0,
        'aircraft_type_B737': 1 if params['ac_type'] == 'B737' else 0,
        'aircraft_type_B777': 1 if params['ac_type'] == 'B777' else 0,
        # เพิ่ม Feature วิศวกรรมพื้นฐาน (ช่วยให้โมเดลคำนวณแม่นขึ้น)
        'weight_x_wind': params['zfw'] * params['wind']
    }
    
    df = pd.DataFrame([data])
    
    # สำคัญ: Reindex ต้องมีคอลัมน์ที่โมเดลต้องการครบถ้วน
    if col_list:
        df = df.reindex(columns=col_list, fill_value=0)
    return df

# --- 6. Prediction & Debug Display ---
if model and model_columns:
    input_df = build_feature_df(final_params, model_columns)
    
    # ส่วน Debug: ไว้เช็คว่าค่า "ขยับ" จริงไหมก่อนเข้า Model
    with st.expander("🔍 Debug: Data sent to Model"):
        st.write("ถ้าเลื่อน Slider แล้วตัวเลขในตารางนี้ต้องเปลี่ยน:")
        st.dataframe(input_df)

    # ทำนายผล
    prediction = model.predict(input_df)[0]

    # แสดงผล Metric
    st.markdown("---")
    st.subheader("2. Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Fuel Burn", f"{prediction:,.2f} kg")
    c2.metric("Selected Aircraft", final_params['ac_type'])
    c3.metric("Source", "File" if input_mode != "Manual (Sidebar Sliders)" else "Manual")

    # --- 7. Risk Index (Gauge) ---
    st.markdown("---")
    st.subheader("3. Safety & Risk Insights")
    
    # คำนวณความเสี่ยงแบบ Simple Logic (ปรับแต่งได้ตามใจ)
    risk_val = 0
    if final_params['wind'] > 20: risk_val += 4
    if final_params['engine'] > 1.05: risk_val += 3
    if final_params['duration'] > 480: risk_val += 2
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_val,
        title = {'text': "Flight Risk Score"},
        gauge = {'axis': {'range': [0, 10]},
                 'steps': [
                     {'range': [0, 3], 'color': "lightgreen"},
                     {'range': [3, 7], 'color': "orange"},
                     {'range': [7, 10], 'color': "red"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ ไม่พบ Model! กรุณาตรวจสอบว่ามีไฟล์ fuel_model.pkl และ model_columns.pkl ในโฟลเดอร์")