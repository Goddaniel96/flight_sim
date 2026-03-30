import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# --- 1. สร้างข้อมูลจำลองแบบสมจริง (Synthetic Data) ---
print("⏳ Generating synthetic flight data...")
np.random.seed(42)
n_samples = 2000

# จำลองค่า Input
zfw = np.random.uniform(50000, 150000, n_samples) # kg
duration = np.random.uniform(60, 600, n_samples)  # min
altitude = np.random.uniform(25000, 41000, n_samples) # ft
headwind = np.random.uniform(-40, 40, n_samples)  # kts
engine_health = np.random.uniform(0.95, 1.05, n_samples) # factor
ac_types = np.random.choice(['A320', 'A330', 'B737', 'B777', 'A350', 'B787'], n_samples)

# --- 2. สร้างสูตรคำนวณค่าน้ำมันจำลอง (สำคัญมาก!) ---
# สูตรนี้กำหนดให้น้ำมันขึ้นอยู่กับ ZFW, Duration, และ Altitude ชัดเจน
# - ZFW ยิ่งเยอะ น้ำมันยิ่งเยอะ (weight = 0.15)
# - Duration ยิ่งนาน น้ำมันยิ่งเยอะ (weight = 50.0)
# - Altitude ยิ่งสูง น้ำมันยิ่งน้อยลงเล็กน้อย (weight = -0.05) - ประหยัดขึ้น
base_fuel = 2000 # ค่าเริ่มต้น
fuel_burn = (
    base_fuel +
    (zfw * 0.15) +           # ZFW effect
    (duration * 50.0) -      # Duration effect
    (altitude * 0.05) +      # Altitude effect (higher is better)
    (headwind * 10.0) +      # Headwind effect
    (zfw * duration * 0.001) + # Interaction
    np.random.normal(0, 500, n_samples) # เพิ่ม Noise เล็กน้อย
)

# จัดการ One-Hot Encoding สำหรับ Aircraft Type
df = pd.DataFrame({
    'zero_fuel_weight_kg': zfw,
    'planned_duration_min': duration,
    'planned_altitude_ft': altitude,
    'avg_headwind_kts': headwind,
    'engine_health_modifier': engine_health,
    'aircraft_type': ac_types,
    'weight_x_wind': zfw * headwind, # เพิ่ม Interaction feature
    'fuel_burn_kg': fuel_burn
})

# แปลง Category เป็น Dummy Columns
df_encoded = pd.get_dummies(df, columns=['aircraft_type'], prefix='aircraft_type', drop_first=True)

# --- 3. เทรนโมเดล Linear Regression ---
print("🧠 Training realistic dummy model...")
X = df_encoded.drop(columns=['fuel_burn_kg'])
y = df_encoded['fuel_burn_kg']

model = LinearRegression()
model.fit(X, y)

# ดูค่า weight ที่โมเดลเรียนรู้ (เพื่อความชัวร์)
print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col}: {coef:.4f}")

# --- 4. บันทึกโมเดลและรายชื่อคอลัมน์ ---
print("\n💾 Saving model to 'fuel_model.pkl'...")
joblib.dump(model, 'fuel_model.pkl')

print("💾 Saving columns to 'model_columns.pkl'...")
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')

print("\n✅ Success! New realistic model created.")