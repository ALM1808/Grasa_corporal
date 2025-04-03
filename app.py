import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import locale
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# Conexión a Google Sheets
import json
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["gspread"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
client = gspread.authorize(creds)
sheet = client.open("Historial_Grasa_Corporal").sheet1

# Inicio de app Streamlit
st.title("Predicción del Porcentaje de Grasa Corporal")

# Forzar el uso del punto como separador decimal
locale.setlocale(locale.LC_NUMERIC, "C")

def normalize_input(value):
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return value

# Cargar modelo y nombres de columnas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

best_model = joblib.load(MODEL_PATH)

with open(FEATURE_NAMES_PATH, "rb") as f:
    feature_names = pickle.load(f)

# Título de la app
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
nombre = st.text_input("👤 Nombre o identificador del usuario", value="usuario1")
if st.button("Predecir Grasa Corporal"):
    # Crear DataFrame de entrada
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
        "Age": [Age],
        "Gender_Male": [1 if Gender == "Male" else 0]
    })

    # Variables derivadas (igual que en entrenamiento)
    input_data["Heart_Rate_Diff"] = input_data["Max_BPM"] - input_data["Resting_BPM"]
    input_data["Calories_per_Hour"] = input_data["Calories_Burned"] / (input_data["Session_Duration (hours)"] + 1e-5)
    input_data["Weight_Height_Ratio"] = input_data["Weight (kg)"] / (input_data["Height (m)"] ** 2)
    input_data["Effort_Ratio"] = input_data["Avg_BPM"] / input_data["Max_BPM"]
    input_data["Hydration_Level"] = input_data["Water_Intake (liters)"] / input_data["Weight (kg)"]
    input_data["Activity_Index"] = input_data["Workout_Frequency (days/week)"] * input_data["Session_Duration (hours)"]
    input_data["Experience_Activity_Ratio"] = input_data["Experience_Level"] / (input_data["Workout_Frequency (days/week)"] + 1)

    st.write("🧾 Datos originales y derivados:", input_data)

    # Filtrar columnas esperadas por el modelo
    try:
        input_model = input_data[feature_names]
    except KeyError as e:
        st.error(f"⚠️ Faltan columnas requeridas por el modelo: {e}")
        st.stop()

    # Predicción
    prediction = best_model.predict(input_model)

    # Mostrar resultado
    st.success(f"✅ El porcentaje estimado de grasa corporal es: {prediction[0]:.2f}%")

    grasa = prediction[0]

    if Gender == "Male":
        if grasa < 10:
            st.info("📉 Por debajo del rango saludable para hombres.")
        elif grasa < 20:
            st.success("✅ Nivel saludable de grasa corporal para hombres.")
        elif grasa < 25:
            st.warning("⚠️ Indicios de sobrepeso según el porcentaje de grasa.")
        else:
            st.error("🚨 Nivel de grasa elevado (obesidad) para hombres.")

    elif Gender == "Female":
        if grasa < 20:
            st.info("📉 Por debajo del rango saludable para mujeres.")
        elif grasa < 30:
            st.success("✅ Nivel saludable de grasa corporal para mujeres.")
        elif grasa < 35:
            st.warning("⚠️ Indicios de sobrepeso según el porcentaje de grasa.")
        else:
            st.error("🚨 Nivel de grasa elevado (obesidad) para mujeres.")
    registro = [
    nombre,
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    round(prediction[0], 2),
    Gender, Age, Weight, Height,
    Max_BPM, Resting_BPM, Avg_BPM,
    Calories_Burned, Session_Duration,
    Water_Intake, Workout_Frequency, Experience_Level
    ]

    # Escribir en la hoja
    st.write("📤 Intentando guardar en Google Sheets...")

    try:
        sheet.append_row(registro)
        st.success("✅ Predicción guardada en Google Sheets.")
    except Exception as e:
        st.error(f"❌ Error al guardar en Google Sheets: {e}")

    # Leer todos los registros
    records = sheet.get_all_records()
    df_historial = pd.DataFrame(records)

    # Filtrar por usuario
    df_usuario = df_historial[df_historial["Nombre"] == nombre]

    if not df_usuario.empty:
        st.subheader("📈 Historial de predicciones")
        st.dataframe(df_usuario)
        
        # Descargar historial
        csv_data = df_usuario.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar historial como CSV",
            data=csv_data,
            file_name=f"historial_{nombre}.csv",
            mime='text/csv',
        )
    else:
        st.info("ℹ️ Aún no hay registros para este usuario.")