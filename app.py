import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Deserción Universitaria", layout="wide")

# Título de la aplicación
st.title("Sistema de Predicción de Deserción Universitaria")
st.markdown("""
Complete el formulario con la información del estudiante para predecir el riesgo de deserción.
""")

# Cargar el modelo y las columnas esperadas
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('pipeline_final_desercion.pkl')
        columnas = joblib.load('columnas_esperadas.pkl')
        return pipeline, columnas
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

pipeline, columnas_esperadas = load_model()

# Justo después de cargar el modelo
st.write(f"Tipo de pipeline cargado: {type(pipeline)}")
if pipeline is not None:
    st.write("📌 Revisión del preprocesador:")
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        for name, transformer, cols in preprocessor.transformers:
            st.write(f"Step: {name}, Tipo de objeto: {type(transformer)}")
    except Exception as e:
        st.error(f"Error al inspeccionar el preprocesador: {e}")


if pipeline is None:
    st.stop()

# Crear formulario para la entrada de datos
with st.form("student_form"):
    # Dividir el formulario en pestañas para mejor organización
    tab1, tab2, tab3, tab4 = st.tabs(["Información Personal", "Datos Académicos", "Información Económica", "Historial Familiar"])
    
    with tab1:
        st.subheader("Información Personal")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=20)
            gender = st.radio("Género", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
            marital_status = st.radio("Estado civil", 
                                     options=["Single", "Divorced", "FactoUnion", "Separated"],
                                     horizontal=True)
            
        with col2:
            displaced = st.checkbox("Desplazado", value=False)
            debtor = st.checkbox("Deudor", value=False)
            tuition_up_to_date = st.checkbox("Matrícula al día", value=True)
            scholarship = st.checkbox("Becado", value=False)
    
    with tab2:
        st.subheader("Datos Académicos")
        col1, col2 = st.columns(2)
        
        with col1:
            application_order = st.slider("Orden de aplicación", 0, 9, 1)
            daytime_attendance = st.radio("Asistencia", options=[1, 0], 
                                        format_func=lambda x: "Diurna" if x == 1 else "Nocturna",
                                        horizontal=True)
            
            prev_qualification_grade = st.number_input("Nota de titulación previa", min_value=0.0, max_value=200.0, value=120.0)
            
            prev_qualification = st.selectbox("Tipo de titulación previa", options=[
                "Secondary Education", "Higher Education", "Technical Education", "Other"
            ])
            
            admission_grade = st.number_input("Nota de admisión", min_value=0.0, max_value=200.0, value=120.0)
            
        with col2:
            application_mode = st.selectbox("Modo de aplicación", options=[
                "Admisión Regular", "Admisión Especial", 
                "Admisión por Ordenanza", "Cambios/Transferencias",
                "Estudiantes Internacionales", "Mayores de 23 años"
            ])
            
            course = st.selectbox("Curso", options=[
                "Agricultural & Environmental Sciences", "Arts & Design",
                "Business & Management", "Communication & Media",
                "Education", "Engineering & Technology",
                "Health Sciences", "Social Sciences"
            ])
            
            # Primer semestre
            st.markdown("**Primer Semestre**")
            col2a, col2b = st.columns(2)
            with col2a:
                units_1sem_eval = st.number_input("Unidades evaluadas (1er sem)", min_value=0, max_value=45, value=5)
            with col2b:
                units_1sem_noeval = st.number_input("Unidades no evaluadas (1er sem)", min_value=0, max_value=12, value=0)
            
            # Segundo semestre
            st.markdown("**Segundo Semestre**")
            col2c, col2d = st.columns(2)
            with col2c:
                units_2sem_credited = st.number_input("Unidades con crédito (2do sem)", min_value=0, max_value=19, value=0)
                units_2sem_enrolled = st.number_input("Unidades matriculadas (2do sem)", min_value=0, max_value=23, value=6)
                units_2sem_eval = st.number_input("Unidades evaluadas (2do sem)", min_value=0, max_value=33, value=6)
            with col2d:
                units_2sem_approved = st.number_input("Unidades aprobadas (2do sem)", min_value=0, max_value=20, value=4)
                units_2sem_grade = st.number_input("Nota media (2do sem)", min_value=0.0, max_value=18.57, value=12.0)
                units_2sem_noeval = st.number_input("Unidades no evaluadas (2do sem)", min_value=0, max_value=12, value=0)
    
    with tab3:
        st.subheader("Información Económica")
        col1, col2 = st.columns(2)
        
        with col1:
            unemployment = st.slider("Tasa de desempleo (%)", min_value=7.6, max_value=16.2, value=10.0)
            inflation = st.slider("Tasa de inflación (%)", min_value=-0.8, max_value=3.7, value=1.5)
        
        with col2:
            gdp = st.slider("GDP", min_value=-4.06, max_value=3.51, value=0.0)
    
    with tab4:
        st.subheader("Historial Familiar")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Madre**")
            mother_qualification = st.radio("Titulación de la madre", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate", "Technical_Education"],
                                          horizontal=True)
            
            mother_occupation = st.selectbox("Ocupación de la madre", options=[
                "Administrative/Clerical", "Skilled Manual Workers",
                "Special Cases", "Technicians/Associate Professionals",
                "Unskilled Workers"
            ])
        
        with col2:
            st.markdown("**Padre**")
            father_qualification = st.radio("Titulación del padre", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate"],
                                          horizontal=True)
            
            father_occupation = st.selectbox("Ocupación del padre", options=[
                "Administrative/Clerical", "Professionals",
                "Skilled Manual Workers", "Special Cases",
                "Technicians/Associate Professionals"
            ])
    
        # Nacionalidad
        nationality = st.selectbox("Nacionalidad", options=[
            "Colombian", "Cuban", "Dutch", "English", "German",
            "Italian", "Lithuanian", "Moldovan", "Mozambican",
            "Portuguese", "Romanian", "Santomean", "Turkish"
        ])
    
    submitted = st.form_submit_button("Predecir Riesgo de Deserción")

# Cuando se envía el formulario
if submitted:
    # Crear un diccionario con todas las columnas esperadas inicializadas a 0
    input_data = {col: 0 for col in columnas_esperadas}
    
    # Actualizar los valores numéricos directos
    input_data.update({
        # Información personal
        'Age at enrollment': age,
        'Gender': gender,
        'Displaced': int(displaced),
        'Debtor': int(debtor),
        'Tuition fees up to date': int(tuition_up_to_date),
        'Scholarship holder': int(scholarship),
        
        # Datos académicos
        'Application order': application_order,
        'Daytime/evening attendance': daytime_attendance,
        'Previous qualification (grade)': prev_qualification_grade,
        'Admission grade': admission_grade,
        
        # Primer semestre
        'Curricular units 1st sem (evaluations)': units_1sem_eval,
        'Curricular units 1st sem (without evaluations)': units_1sem_noeval,
        
        # Segundo semestre
        'Curricular units 2nd sem (credited)': units_2sem_credited,
        'Curricular units 2nd sem (enrolled)': units_2sem_enrolled,
        'Curricular units 2nd sem (evaluations)': units_2sem_eval,
        'Curricular units 2nd sem (approved)': units_2sem_approved,
        'Curricular units 2nd sem (grade)': units_2sem_grade,
        'Curricular units 2nd sem (without evaluations)': units_2sem_noeval,
        
        # Información económica
        'Unemployment rate': unemployment,
        'Inflation rate': inflation,
        'GDP': gdp,
    })
    
    # Activar variables categóricas usando los nombres exactos del modelo
    # Estado civil
    marital_col = f'Marital status_{marital_status}'
    if marital_col in columnas_esperadas:
        input_data[marital_col] = 1
    
    # Modo de aplicación - usar nombres exactos del modelo
    app_mode_map = {
        "Admisión Regular": "Admisión Regular",
        "Admisión Especial": "Admisión Especial", 
        "Admisión por Ordenanza": "Admisión por Ordenanza",
        "Cambios/Transferencias": "Cambios/Transferencias",
        "Estudiantes Internacionales": "Estudiantes Internacionales",
        "Mayores de 23 años": "Mayores de 23 años"
    }
    app_mode_col = f'Application mode_{app_mode_map[application_mode]}'
    if app_mode_col in columnas_esperadas:
        input_data[app_mode_col] = 1
    
    # Curso
    course_col = f'Course_{course}'
    if course_col in columnas_esperadas:
        input_data[course_col] = 1
    
    # Titulación previa
    prev_qual_col = f'Previous qualification_{prev_qualification}'
    if prev_qual_col in columnas_esperadas:
        input_data[prev_qual_col] = 1
    
    # Nacionalidad
    nationality_col = f'Nacionality_{nationality}'
    if nationality_col in columnas_esperadas:
        input_data[nationality_col] = 1
    
    # Titulación de la madre
    mother_qual_col = f"Mother's qualification_{mother_qualification}"
    if mother_qual_col in columnas_esperadas:
        input_data[mother_qual_col] = 1
    
    # Ocupación de la madre - corregir nombres con espacios
    mother_occ_formatted = mother_occupation.replace(" ", "_").replace("/", "/")
    mother_occ_col = f"Mother's occupation_{mother_occ_formatted}"
    if mother_occ_col in columnas_esperadas:
        input_data[mother_occ_col] = 1
    
    # Titulación del padre
    father_qual_col = f"Father's qualification_{father_qualification}"
    if father_qual_col in columnas_esperadas:
        input_data[father_qual_col] = 1
    
    # Ocupación del padre - corregir nombres con espacios
    father_occ_formatted = father_occupation.replace(" ", "_").replace("/", "/")
    father_occ_col = f"Father's occupation_{father_occ_formatted}"
    if father_occ_col in columnas_esperadas:
        input_data[father_occ_col] = 1
    
    # Convertir a DataFrame manteniendo el orden exacto de las columnas esperadas
    input_df = pd.DataFrame([input_data])[columnas_esperadas]
    
    # Hacer la predicción
    try:
        input_df = input_df.astype(np.float32)

        prediction_proba = pipeline.predict_proba(input_df)[0][1]  # Probabilidad de deserción
        prediction_percent = round(prediction_proba * 100, 2)
        
        # Mostrar resultados
        st.subheader("Resultado de la Predicción")
        
        # Barra de progreso y métrica
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad de Deserción", f"{prediction_percent}%")
        with col2:
            st.progress(prediction_proba)
        
        # Interpretación
        if prediction_proba > 0.7:
            st.error("🚨 Alto riesgo de deserción. Se recomienda intervención inmediata.")
        elif prediction_proba > 0.4:
            st.warning("⚠️ Riesgo moderado de deserción. Se sugiere monitoreo cercano.")
        else:
            st.success("✅ Bajo riesgo de deserción.")
        
        # Mostrar información adicional
        with st.expander("Ver detalles de la predicción"):
            st.write("**Factores de riesgo identificados:**")
            
            risk_factors = []
            if prediction_proba > 0.5:
                if units_2sem_grade < 10:
                    risk_factors.append("- Nota promedio baja en segundo semestre")
                if units_2sem_approved < units_2sem_eval * 0.7:
                    risk_factors.append("- Baja tasa de aprobación en segundo semestre")
                if not tuition_up_to_date:
                    risk_factors.append("- Matrícula no está al día")
                if debtor:
                    risk_factors.append("- Estudiante con deudas")
                if unemployment > 12:
                    risk_factors.append("- Alta tasa de desempleo en el contexto")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.write("No se identificaron factores de riesgo significativos.")
        
        # Mostrar los datos técnicos (opcional)
        with st.expander("Ver datos técnicos enviados al modelo"):
            # Mostrar solo las columnas con valores no cero para mayor claridad
            non_zero_data = {k: v for k, v in input_data.items() if v != 0}
            st.json(non_zero_data)
            
    except Exception as e:
        st.error(f"Error al hacer la predicción: {str(e)}")
        st.write("Columnas esperadas por el modelo:")
        st.write(columnas_esperadas)
        st.write("Columnas enviadas:")
        st.write(list(input_df.columns))

# Información adicional en el sidebar
st.sidebar.markdown("""
### Instrucciones:
1. Complete todas las pestañas del formulario.
2. Revise que los datos sean correctos.
3. Haga clic en **Predecir Riesgo de Deserción**.
4. Interprete los resultados según el nivel de riesgo.

### Notas:
- Los campos booleanos se convierten automáticamente (0=False, 1=True)
- Para variables categóricas, seleccione solo una opción
- El modelo fue entrenado con datos balanceados usando SMOTE
- Utiliza un pipeline con XGBoost optimizado mediante búsqueda bayesiana

### Métricas del modelo:
- F1-Score: ~0.91
- Validación cruzada de 10 pliegues
- Optimización de hiperparámetros con BayesSearchCV
""")

# Footer
st.markdown("---")
st.markdown("*Sistema desarrollado para predecir la deserción universitaria usando técnicas de Machine Learning*")