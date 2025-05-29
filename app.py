import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar el modelo con manejo de errores
@st.cache_resource
def load_model():
    try:
        # Intentar cargar el pipeline completo
        if os.path.exists('pipeline_final_desercion1.pkl'):
            model = joblib.load('pipeline_final_desercion1.pkl')
            st.success("✅ Modelo cargado exitosamente")
            return model, "pipeline"
        else:
            st.error("❌ No se encontró el archivo del modelo")
            return None, None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None, None

# Función para validar y preparar los datos de entrada
def prepare_input_data(input_dict):
    """
    Prepara los datos de entrada asegurando que tengan la estructura correcta
    """
    try:
        # Crear DataFrame con los datos de entrada
        df = pd.DataFrame([input_dict])
        
        # Verificar que todas las columnas necesarias estén presentes
        expected_columns = get_expected_columns()
        
        # Agregar columnas faltantes con valor 0
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reordenar columnas según el orden esperado
        df = df[expected_columns]
        
        # Convertir tipos de datos
        df = convert_data_types(df)
        
        return df
    
    except Exception as e:
        st.error(f"Error al preparar los datos: {str(e)}")
        return None
    
def convert_data_types(df):
    """
    Convierte los tipos de datos según las especificaciones del modelo
    """

    # Variables enteras
    variables_int = [
        'Application order', 'Daytime/evening attendance', 'Displaced',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'Age at enrollment', 'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (without evaluations)', 
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (without evaluations)'
    ]
    
    # Variables flotantes
    variables_float = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 2nd sem (grade)', 'Unemployment rate',
        'Inflation rate', 'GDP'
    ]
    
    # Variables adicionales que se pasan como passthrough y deben ser enteras (dummies)
    variables_passthrough = [
        'Marital status_Divorced', 'Marital status_FactoUnion', 'Marital status_Separated', 'Marital status_Single',
        'Application mode_Admisión Especial', 'Application mode_Admisión Regular', 'Application mode_Admisión por Ordenanza',
        'Application mode_Cambios/Transferencias', 'Application mode_Estudiantes Internacionales', 'Application mode_Mayores de 23 años',
        'Course_Agricultural & Environmental Sciences', 'Course_Arts & Design', 'Course_Business & Management',
        'Course_Communication & Media', 'Course_Education', 'Course_Engineering & Technology', 'Course_Health Sciences',
        'Course_Social Sciences', 'Previous qualification_Higher Education', 'Previous qualification_Other',
        'Previous qualification_Secondary Education', 'Previous qualification_Technical Education', 'Nacionality_Colombian',
        'Nacionality_Cuban', 'Nacionality_Dutch', 'Nacionality_English', 'Nacionality_German', 'Nacionality_Italian',
        'Nacionality_Lithuanian', 'Nacionality_Moldovan', 'Nacionality_Mozambican', 'Nacionality_Portuguese',
        'Nacionality_Romanian', 'Nacionality_Santomean', 'Nacionality_Turkish',
        "Mother's qualification_Basic_or_Secondary", "Mother's qualification_Other_or_Unknown",
        "Mother's qualification_Postgraduate", "Mother's qualification_Technical_Education",
        "Father's qualification_Basic_or_Secondary", "Father's qualification_Other_or_Unknown",
        "Father's qualification_Postgraduate", "Mother's occupation_Administrative/Clerical",
        "Mother's occupation_Skilled Manual Workers", "Mother's occupation_Special Cases",
        "Mother's occupation_Technicians/Associate Professionals", "Mother's occupation_Unskilled Workers",
        "Father's occupation_Administrative/Clerical", "Father's occupation_Professionals",
        "Father's occupation_Skilled Manual Workers", "Father's occupation_Special Cases",
        "Father's occupation_Technicians/Associate Professionals"
    ]

    # Convertir variables enteras
    for col in variables_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Convertir variables flotantes
    for col in variables_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')

    # Convertir variables passthrough (dummies) a int
    for col in variables_passthrough:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    return df


def get_expected_columns():
    """
    Retorna la lista de columnas esperadas por el modelo
    """
    return [
        'Application order', 'Daytime/evening attendance', 'Previous qualification (grade)',
        'Admission grade', 'Displaced', 'Debtor', 'Tuition fees up to date', 'Gender',
        'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
        'Inflation rate', 'GDP', 'Marital status_Divorced', 'Marital status_FactoUnion',
        'Marital status_Separated', 'Marital status_Single', 'Application mode_Admisión Especial',
        'Application mode_Admisión Regular', 'Application mode_Admisión por Ordenanza',
        'Application mode_Cambios/Transferencias', 'Application mode_Estudiantes Internacionales',
        'Application mode_Mayores de 23 años', 'Course_Agricultural & Environmental Sciences',
        'Course_Arts & Design', 'Course_Business & Management', 'Course_Communication & Media',
        'Course_Education', 'Course_Engineering & Technology', 'Course_Health Sciences',
        'Course_Social Sciences', 'Previous qualification_Higher Education',
        'Previous qualification_Other', 'Previous qualification_Secondary Education',
        'Previous qualification_Technical Education', 'Nacionality_Colombian',
        'Nacionality_Cuban', 'Nacionality_Dutch', 'Nacionality_English', 'Nacionality_German',
        'Nacionality_Italian', 'Nacionality_Lithuanian', 'Nacionality_Moldovan',
        'Nacionality_Mozambican', 'Nacionality_Portuguese', 'Nacionality_Romanian',
        'Nacionality_Santomean', 'Nacionality_Turkish', "Mother's qualification_Basic_or_Secondary",
        "Mother's qualification_Other_or_Unknown", "Mother's qualification_Postgraduate",
        "Mother's qualification_Technical_Education", "Father's qualification_Basic_or_Secondary",
        "Father's qualification_Other_or_Unknown", "Father's qualification_Postgraduate",
        "Mother's occupation_Administrative/Clerical", "Mother's occupation_Skilled Manual Workers",
        "Mother's occupation_Special Cases", "Mother's occupation_Technicians/Associate Professionals",
        "Mother's occupation_Unskilled Workers", "Father's occupation_Administrative/Clerical",
        "Father's occupation_Professionals", "Father's occupation_Skilled Manual Workers",
        "Father's occupation_Special Cases", "Father's occupation_Technicians/Associate Professionals"
    ]

def main():
    st.title("🎓 Predictor de Deserción Estudiantil")
    st.markdown("---")
    
    # Cargar modelo
    model, model_type = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifique que el archivo 'pipeline_final_desercion.pkl' esté disponible.")
        return
    
    # Sidebar para información
    with st.sidebar:
        st.header("ℹ️ Información del Modelo")
        st.write("**Algoritmo:** XGBoost")
        st.write("**Métricas de rendimiento:**")
        st.write("- F1-Score: ~0.91")
        st.write("- Precisión: Alta")
        st.write("- Recall: Alto")
        
        st.markdown("---")
        st.write("**Instrucciones:**")
        st.write("1. Complete los campos del formulario")
        st.write("2. Haga clic en 'Predecir'")
        st.write("3. Obtenga la predicción y probabilidad")
    
    # Formulario principal
    with st.form("prediction_form"):
        st.header("📝 Datos del Estudiante")
        
        # Organizamos en columnas para mejor UX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Información Personal")
            age = st.number_input("Edad al inscribirse", min_value=16, max_value=70, value=20)
            gender = st.selectbox("Género", ["Masculino", "Femenino"])
            marital_status = st.selectbox("Estado civil", 
                ["Single", "Divorced", "FactoUnion", "Separated"])
            
            st.subheader("Información Académica")
            application_order = st.number_input("Orden de aplicación", min_value=1, max_value=10, value=1)
            daytime_evening = st.selectbox("Asistencia", ["Diurna", "Nocturna"])
            admission_grade = st.number_input("Calificación de admisión", 
                min_value=0.0, max_value=200.0, value=120.0, step=0.1)
            previous_qualification_grade = st.number_input("Calificación previa", 
                min_value=0.0, max_value=200.0, value=120.0, step=0.1)
        
        with col2:
            st.subheader("Información Económica")
            displaced = st.selectbox("Desplazado", ["No", "Sí"])
            debtor = st.selectbox("Deudor", ["No", "Sí"])
            tuition_up_to_date = st.selectbox("Matrícula al día", ["No", "Sí"])
            scholarship = st.selectbox("Becario", ["No", "Sí"])
            
            st.subheader("Contexto Socioeconómico")
            unemployment_rate = st.number_input("Tasa de desempleo (%)", 
                min_value=0.0, max_value=30.0, value=10.0, step=0.1)
            inflation_rate = st.number_input("Tasa de inflación (%)", 
                min_value=-5.0, max_value=20.0, value=2.0, step=0.1)
            gdp = st.number_input("PIB", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
        
        with col3:
            st.subheader("Rendimiento Académico")
            sem1_evaluations = st.number_input("Evaluaciones 1er semestre", 
                min_value=0, max_value=30, value=6)
            sem1_without_eval = st.number_input("Sin evaluación 1er semestre", 
                min_value=0, max_value=10, value=0)
            
            sem2_credited = st.number_input("Materias acreditadas 2do semestre", 
                min_value=0, max_value=30, value=0)
            sem2_enrolled = st.number_input("Materias inscritas 2do semestre", 
                min_value=0, max_value=30, value=6)
            sem2_evaluations = st.number_input("Evaluaciones 2do semestre", 
                min_value=0, max_value=30, value=6)
            sem2_approved = st.number_input("Materias aprobadas 2do semestre", 
                min_value=0, max_value=30, value=5)
            sem2_grade = st.number_input("Calificación 2do semestre", 
                min_value=0.0, max_value=20.0, value=13.0, step=0.1)
            sem2_without_eval = st.number_input("Sin evaluación 2do semestre", 
                min_value=0, max_value=10, value=0)
        
        # Selectboxes para variables categóricas principales
        st.subheader("Información Adicional")
        col4, col5 = st.columns(2)
        
        with col4:
            application_mode = st.selectbox("Modo de aplicación", [
                "Admisión Regular", "Admisión Especial", "Admisión por Ordenanza",
                "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"
            ])
            
            course = st.selectbox("Curso", [
                "Engineering & Technology", "Business & Management", "Health Sciences",
                "Education", "Social Sciences", "Arts & Design", 
                "Agricultural & Environmental Sciences", "Communication & Media"
            ])
        
        with col5:
            previous_qualification = st.selectbox("Calificación previa", [
                "Secondary Education", "Higher Education", "Technical Education", "Other"
            ])
            
            nationality = st.selectbox("Nacionalidad", [
                "Portuguese", "Colombian", "German", "Italian", "English", 
                "Dutch", "Turkish", "Cuban", "Lithuanian", "Moldovan",
                "Mozambican", "Romanian", "Santomean"
            ])
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Deserción", use_container_width=True)
        
        if submitted:
            # Crear diccionario con los datos
            input_data = create_input_dictionary(
                age, gender, marital_status, application_order, daytime_evening,
                admission_grade, previous_qualification_grade, displaced, debtor,
                tuition_up_to_date, scholarship, unemployment_rate, inflation_rate,
                gdp, sem1_evaluations, sem1_without_eval, sem2_credited,
                sem2_enrolled, sem2_evaluations, sem2_approved, sem2_grade,
                sem2_without_eval, application_mode, course, previous_qualification,
                nationality
            )
            
            # Preparar datos
            df_input = prepare_input_data(input_data)
            
            if df_input is not None:
                try:
                    # Realizar predicción
                    prediction = model.predict(df_input)[0]
                    probability = model.predict_proba(df_input)[0]
                    
                    # Mostrar resultados
                    st.markdown("---")
                    st.header("📊 Resultados de la Predicción")
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        if prediction == 1:
                            st.error("🚨 **ALTO RIESGO DE DESERCIÓN**")
                            st.write(f"Probabilidad de deserción: **{probability[1]:.2%}**")
                        else:
                            st.success("✅ **BAJO RIESGO DE DESERCIÓN**")
                            st.write(f"Probabilidad de permanecer: **{probability[0]:.2%}**")
                    
                    with col_result2:
                        st.write("**Distribución de Probabilidades:**")
                        prob_df = pd.DataFrame({
                            'Clase': ['Permanece', 'Deserta'],
                            'Probabilidad': [probability[0], probability[1]]
                        })
                        st.bar_chart(prob_df.set_index('Clase'))
                    
                    # Recomendaciones
                    if prediction == 1:
                        st.warning("### 💡 Recomendaciones de Intervención")
                        st.write("- Considerar programas de apoyo académico")
                        st.write("- Evaluar necesidades de apoyo financiero")
                        st.write("- Implementar seguimiento personalizado")
                        st.write("- Conectar con servicios de bienestar estudiantil")
                
                except Exception as e:
                    st.error(f"Error al realizar la predicción: {str(e)}")
                    st.write("Verifique que todos los campos estén completados correctamente.")

def create_input_dictionary(age, gender, marital_status, application_order, daytime_evening,
                          admission_grade, previous_qualification_grade, displaced, debtor,
                          tuition_up_to_date, scholarship, unemployment_rate, inflation_rate,
                          gdp, sem1_evaluations, sem1_without_eval, sem2_credited,
                          sem2_enrolled, sem2_evaluations, sem2_approved, sem2_grade,
                          sem2_without_eval, application_mode, course, previous_qualification,
                          nationality):
    """
    Crea el diccionario de entrada con todas las variables necesarias
    """
    # Inicializar todas las columnas con 0
    input_dict = {col: 0 for col in get_expected_columns()}
    
    # Asignar valores básicos
    input_dict.update({
        'Application order': application_order,
        'Daytime/evening attendance': 1 if daytime_evening == "Diurna" else 0,
        'Previous qualification (grade)': previous_qualification_grade,
        'Admission grade': admission_grade,
        'Displaced': 1 if displaced == "Sí" else 0,
        'Debtor': 1 if debtor == "Sí" else 0,
        'Tuition fees up to date': 1 if tuition_up_to_date == "Sí" else 0,
        'Gender': 1 if gender == "Masculino" else 0,
        'Scholarship holder': 1 if scholarship == "Sí" else 0,
        'Age at enrollment': age,
        'Curricular units 1st sem (evaluations)': sem1_evaluations,
        'Curricular units 1st sem (without evaluations)': sem1_without_eval,
        'Curricular units 2nd sem (credited)': sem2_credited,
        'Curricular units 2nd sem (enrolled)': sem2_enrolled,
        'Curricular units 2nd sem (evaluations)': sem2_evaluations,
        'Curricular units 2nd sem (approved)': sem2_approved,
        'Curricular units 2nd sem (grade)': sem2_grade,
        'Curricular units 2nd sem (without evaluations)': sem2_without_eval,
        'Unemployment rate': unemployment_rate,
        'Inflation rate': inflation_rate,
        'GDP': gdp
    })
    
    # Variables categóricas - Estado civil
    input_dict[f'Marital status_{marital_status}'] = 1
    
    # Modo de aplicación
    input_dict[f'Application mode_{application_mode}'] = 1
    
    # Curso
    input_dict[f'Course_{course}'] = 1
    
    # Calificación previa
    input_dict[f'Previous qualification_{previous_qualification}'] = 1
    
    # Nacionalidad
    input_dict[f'Nacionality_{nationality}'] = 1
    
    return input_dict

if __name__ == "__main__":
    main()