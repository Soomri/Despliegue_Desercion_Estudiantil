import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y columnas esperadas
pipeline = joblib.load("pipeline_final_desercion.pkl")
columnas_esperadas = joblib.load("columnas_esperadas.pkl")

st.title("Predicción de Deserción Estudiantil")

st.write("""
Este modelo predice la probabilidad de deserción de un estudiante, basado en características académicas y socioeconómicas.
""")

# Opción de entrada manual o por archivo
opcion = st.radio("Selecciona el tipo de entrada:", ["Manual", "Archivo CSV"])

if opcion == "Manual":
    entrada = {}
    for col in columnas_esperadas:
        entrada[col] = st.text_input(col, value="0")

    if st.button("Predecir"):
        try:
            entrada_df = pd.DataFrame([entrada])
            entrada_df = entrada_df.astype(float)
            pred = pipeline.predict(entrada_df)[0]
            prob = pipeline.predict_proba(entrada_df)[0][1]
            st.success(f"Predicción: {'Deserta' if pred == 1 else 'No Deserta'} (Probabilidad: {prob:.2f})")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

else:
    archivo = st.file_uploader("Sube un archivo CSV con los datos:", type="csv")
    if archivo is not None:
        df = pd.read_csv(archivo)
        if set(columnas_esperadas).issubset(df.columns):
            df = df[columnas_esperadas]
            predicciones = pipeline.predict(df)
            probabilidades = pipeline.predict_proba(df)[:, 1]
            resultado = df.copy()
            resultado["Predicción"] = predicciones
            resultado["Probabilidad de Deserción"] = probabilidades
            st.write(resultado)
        else:
            st.error("Las columnas del archivo no coinciden con las esperadas.")

