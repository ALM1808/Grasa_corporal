import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import locale

# Forzar el uso del punto como separador decimal
locale.setlocale(locale.LC_NUMERIC, "C")

def normalize_input(value):
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return value  # Devuelve el valor original si no es convertible

# Cargar los modelos y preprocesadores
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")


#scaler = joblib.load(SCALER_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
best_model = joblib.load(MODEL_PATH)

# Configurar la aplicación
st.title("Predicción del Porcentaje de Grasa Corporal")
st.write("Ingrese los datos del usuario para obtener una predicción de su porcentaje de grasa corporal.")

# Entrada de datos del usuario
col1, col2 = st.columns(2)

with col1:
    Max_BPM = normalize_input(st.number_input("Frecuencia Cardíaca Máxima", min_value=60.0, max_value=220.0, value=150.0, format="%.2f"))
    Resting_BPM = normalize_input(st.number_input("Frecuencia Cardíaca en Reposo", min_value=40.0, max_value=100.0, value=70.0, format="%.2f"))
    Calories_Burned = normalize_input(st.number_input("Calorías Quemadas", min_value=0.0, value=300.0, format="%.2f"))
    Session_Duration = normalize_input(st.number_input("Duración de la Sesión (horas)", min_value=0.1, value=1.0, format="%.2f"))
    Weight = normalize_input(st.number_input("Peso (kg)", min_value=30.0, value=70.0, format="%.2f"))
    Height = normalize_input(st.number_input("Altura (m)", min_value=1.0, value=1.75, format="%.2f"))
    Avg_BPM = normalize_input(st.number_input("Frecuencia Cardíaca Promedio", min_value=60.0, max_value=200.0, value=120.0, format="%.2f"))

with col2:
    Water_Intake = normalize_input(st.number_input("Consumo de Agua (litros)", min_value=0.0, value=2.0, format="%.2f"))
    Workout_Frequency = int(st.number_input("Frecuencia de Entrenamiento (días/semana)", min_value=0, max_value=7, value=3, step=1))
    Experience_Level = int(st.number_input("Nivel de Experiencia (años)", min_value=0, value=2, step=1))
    Gender = st.selectbox("Género", ["Male", "Female"])
    Age = int(st.number_input("Edad", min_value=10, max_value=100, value=30, step=1))


# Botón para predecir
if st.button("Predecir Grasa Corporal"):
    # 1. Crear DataFrame base con inputs del usuario
    input_data = pd.DataFrame({
        "Max_BPM": [Max_BPM],
        "Resting_BPM": [Resting_BPM],
        "Calories_Burned": [Calories_Burned],
        "Session_Duration (hours)": [Session_Duration],
        "Weight (kg)": [Weight],
        "Height (m)": [Height],
        "Avg_BPM": [Avg_BPM],
        "Water_Intake (liters)": [Water_Intake],
        "Workout_Frequency (days/week)": [Workout_Frequency],
        "Experience_Level": [Experience_Level],
        "Gender": [Gender],
        "Age": [Age]
    })

    # 2. Agregar variables derivadas (igual que en el entrenamiento)
    input_data["BMI"] = input_data["Weight (kg)"] / (input_data["Height (m)"] ** 2)
    input_data["Heart_Rate_Diff"] = input_data["Max_BPM"] - input_data["Resting_BPM"]
    input_data["Calories_per_Hour"] = input_data["Calories_Burned"] / (input_data["Session_Duration (hours)"] + 1e-5)
    input_data["Weight_Height_Ratio"] = input_data["Weight (kg)"] / (input_data["Height (m)"] ** 2)
    input_data["Effort_Ratio"] = input_data["Avg_BPM"] / input_data["Max_BPM"]
    input_data["Hydration_Level"] = input_data["Water_Intake (liters)"] / input_data["Weight (kg)"]
    input_data["Activity_Index"] = input_data["Workout_Frequency (days/week)"] * input_data["Session_Duration (hours)"]
    input_data["Experience_Activity_Ratio"] = input_data["Experience_Level"] / (input_data["Workout_Frequency (days/week)"] + 1)

    st.write("🧾 Datos originales y derivados:", input_data)

    # 3. Aplicar preprocesamiento (OneHot + Scaling)
    try:
        input_transformed = preprocessor.transform(input_data)
        input_transformed = pd.DataFrame(input_transformed)
    except ValueError as e:
        st.error(f"Error en el preprocesamiento: {e}")
        st.stop()

    # 4. Reordenar columnas según las usadas en el modelo
    FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    input_transformed = input_transformed.reindex(columns=feature_names, fill_value=0)

    # 5. Hacer predicción
    prediction = best_model.predict(input_transformed)

    # 6. Mostrar resultado
    st.success(f"✅ El porcentaje estimado de grasa corporal es: {prediction[0]:.2f}%")
    