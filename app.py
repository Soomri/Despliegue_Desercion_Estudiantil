# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para Predicción de Deserción Estudiantil
Modelo: XGBoost optimizado con Optimización Bayesiana
Basado en metodología CRISP-DM
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Predicción Deserción Estudiantil",
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title('🎓 Predicción de Deserción Estudiantil')
st.markdown("### Modelo XGBoost con Optimización Bayesiana")

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Cargar el pipeline completo
        pipeline = joblib.load('pipeline_final_desercion.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

# Función para obtener las categorías exactas del entrenamiento
@st.cache_data
def get_training_categories():
    categories = {
        'marital_status': ['Divorced', 'FactoUnion', 'Separated', 'Single'],
        'application_mode': [
            'Admisión Especial', 'Admisión Regular', 'Admisión por Ordenanza',
            'Cambios/Transferencias', 'Estudiantes Internacionales', 'Mayores de 23 años'
        ],
        'course': [
            'Agricultural & Environmental Sciences', 'Arts & Design', 'Business & Management',
            'Communication & Media', 'Education', 'Engineering & Technology',
            'Health Sciences', 'Social Sciences'
        ],
        'previous_qualification': [
            'Higher Education', 'Other', 'Secondary Education', 'Technical Education'
        ],
        'nationality': [
            'Colombian', 'Cuban', 'Dutch', 'English', 'German', 'Italian',
            'Lithuanian', 'Moldovan', 'Mozambican', 'Portuguese', 'Romanian',
            'Santomean', 'Turkish'
        ],
        'mother_qualification': [
            'Basic_or_Secondary', 'Other_or_Unknown', 'Postgraduate', 'Technical_Education'
        ],
        'father_qualification': [
            'Basic_or_Secondary', 'Other_or_Unknown', 'Postgraduate', 'Technical_Education'
        ],
        'mother_occupation': [
            'Administrative/Clerical', 'Skilled Manual Workers', 'Special Cases',
            'Technicians/Associate Professionals', 'Unskilled Workers'
        ],
        'father_occupation': [
            'Administrative/Clerical', 'Professionals', 'Skilled Manual Workers',
            'Special Cases', 'Technicians/Associate Professionals'
        ]
    }
    return categories

# Función para crear DataFrame con todas las columnas esperadas
def create_complete_dataframe(user_data, categories):
    """Crear DataFrame con todas las columnas dummies esperadas por el modelo"""
    
    # Crear DataFrame base con datos numéricos
    df = pd.DataFrame([{
        'Application order': user_data['application_order'],
        'Daytime/evening attendance': user_data['daytime_evening'],
        'Previous qualification (grade)': user_data['previous_qualification_grade'],
        'Admission grade': user_data['admission_grade'],
        'Displaced': user_data['displaced'],
        'Debtor': user_data['debtor'],
        'Tuition fees up to date': user_data['tuition_fees'],
        'Gender': user_data['gender'],
        'Scholarship holder': user_data['scholarship'],
        'Age at enrollment': user_data['age_enrollment'],
        'Curricular units 1st sem (evaluations)': user_data['cu_1st_evaluations'],
        'Curricular units 1st sem (without evaluations)': user_data['cu_1st_without_eval'],
        'Curricular units 2nd sem (credited)': user_data['cu_2nd_credited'],
        'Curricular units 2nd sem (enrolled)': user_data['cu_2nd_enrolled'],
        'Curricular units 2nd sem (evaluations)': user_data['cu_2nd_evaluations'],
        'Curricular units 2nd sem (approved)': user_data['cu_2nd_approved'],
        'Curricular units 2nd sem (grade)': user_data['cu_2nd_grade'],
        'Curricular units 2nd sem (without evaluations)': user_data['cu_2nd_without_eval'],
        'Unemployment rate': user_data['unemployment_rate'],
        'Inflation rate': user_data['inflation_rate'],
        'GDP': user_data['gdp']
    }])
    
    # Agregar todas las variables dummy (inicializadas en 0)
    dummy_columns = [
        'Marital status_Divorced', 'Marital status_FactoUnion', 'Marital status_Separated', 'Marital status_Single',
        'Application mode_Admisión Especial', 'Application mode_Admisión Regular', 'Application mode_Admisión por Ordenanza',
        'Application mode_Cambios/Transferencias', 'Application mode_Estudiantes Internacionales', 'Application mode_Mayores de 23 años',
        'Course_Agricultural & Environmental Sciences', 'Course_Arts & Design', 'Course_Business & Management',
        'Course_Communication & Media', 'Course_Education', 'Course_Engineering & Technology',
        'Course_Health Sciences', 'Course_Social Sciences',
        'Previous qualification_Higher Education', 'Previous qualification_Other',
        'Previous qualification_Secondary Education', 'Previous qualification_Technical Education',
        'Nacionality_Colombian', 'Nacionality_Cuban', 'Nacionality_Dutch', 'Nacionality_English',
        'Nacionality_German', 'Nacionality_Italian', 'Nacionality_Lithuanian', 'Nacionality_Moldovan',
        'Nacionality_Mozambican', 'Nacionality_Portuguese', 'Nacionality_Romanian', 'Nacionality_Santomean', 'Nacionality_Turkish',
        "Mother's qualification_Basic_or_Secondary", "Mother's qualification_Other_or_Unknown",
        "Mother's qualification_Postgraduate", "Mother's qualification_Technical_Education",
        "Father's qualification_Basic_or_Secondary", "Father's qualification_Other_or_Unknown",
        "Father's qualification_Postgraduate",
        "Mother's occupation_Administrative/Clerical", "Mother's occupation_Skilled Manual Workers",
        "Mother's occupation_Special Cases", "Mother's occupation_Technicians/Associate Professionals",
        "Mother's occupation_Unskilled Workers",
        "Father's occupation_Administrative/Clerical", "Father's occupation_Professionals",
        "Father's occupation_Skilled Manual Workers", "Father's occupation_Special Cases",
        "Father's occupation_Technicians/Associate Professionals"
    ]
    
    # Inicializar todas las dummies en 0
    for col in dummy_columns:
        df[col] = 0
    
    # Activar las dummies correspondientes a las selecciones del usuario
    df[f'Marital status_{user_data["marital_status"]}'] = 1
    df[f'Application mode_{user_data["application_mode"]}'] = 1
    df[f'Course_{user_data["course"]}'] = 1
    df[f'Previous qualification_{user_data["previous_qualification"]}'] = 1
    df[f'Nacionality_{user_data["nationality"]}'] = 1
    df[f"Mother's qualification_{user_data['mother_qualification']}"] = 1
    df[f"Father's qualification_{user_data['father_qualification']}"] = 1
    df[f"Mother's occupation_{user_data['mother_occupation']}"] = 1
    df[f"Father's occupation_{user_data['father_occupation']}"] = 1
    
    return df

# Cargar modelo
pipeline = load_model()

if pipeline is not None:
    st.success("✅ Pipeline cargado exitosamente")

    # Sidebar para información
    st.sidebar.header("ℹ️ Información del Modelo")
    st.sidebar.info("""
    **Modelo:** XGBoost Classifier
    
    **Optimización:** Bayesian Optimization
    
    **Métricas de Performance:**
    - F1-Score: ~0.91
    - Accuracy: ~0.89
    - Precision: ~0.88
    - Recall: ~0.94
    
    **Preprocesamiento:**
    - Variables numéricas: Normalizadas (MinMaxScaler)
    - Variables categóricas: One-Hot Encoding
    - Balanceo: SMOTE aplicado
    """)

    # Obtener categorías del entrenamiento
    categories = get_training_categories()

    # Crear formulario de entrada
    st.header("📊 Ingrese los datos del estudiante")

    # Información personal y académica
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Información Personal")
        
        age_enrollment = st.number_input(
            'Edad al momento de inscripción',
            min_value=16,
            max_value=70,
            value=20,
            help="Edad del estudiante cuando se inscribió"
        )
        
        gender = st.selectbox(
            'Género',
            [0, 1],
            format_func=lambda x: 'Femenino' if x == 0 else 'Masculino',
            help="Género del estudiante"
        )
        
        marital_status = st.selectbox(
            'Estado Civil',
            categories['marital_status'],
            help="Estado civil del estudiante"
        )
        
        displaced = st.selectbox(
            '¿Desplazado?',
            [0, 1],
            format_func=lambda x: 'No' if x == 0 else 'Sí',
            help="¿Es una persona en situación de desplazamiento?"
        )
        
        nationality = st.selectbox(
            'Nacionalidad',
            categories['nationality'],
            index=0,  # Por defecto Colombian
            help="Nacionalidad del estudiante"
        )

    with col2:
        st.subheader("🎓 Información Académica")
        
        application_mode = st.selectbox(
            'Modo de Aplicación',
            categories['application_mode'],
            index=1,  # Por defecto Admisión Regular
            help="Forma de ingreso a la institución"
        )
        
        course = st.selectbox(
            'Programa de Estudio',
            categories['course'],
            help="Programa académico matriculado"
        )
        
        daytime_evening = st.selectbox(
            'Jornada',
            [0, 1],
            format_func=lambda x: 'Diurna' if x == 1 else 'Nocturna',
            help="Jornada de estudio"
        )
        
        previous_qualification = st.selectbox(
            'Calificación Previa',
            categories['previous_qualification'],
            help="Tipo de educación previa"
        )

    # Información económica y familiar
    st.subheader("💰 Información Económica y Familiar")
    col3, col4, col5 = st.columns(3)

    with col3:
        debtor = st.selectbox(
            '¿Deudor?',
            [0, 1],
            format_func=lambda x: 'No' if x == 0 else 'Sí',
            help="¿Tiene deudas pendientes?"
        )
        
        tuition_fees = st.selectbox(
            '¿Matrícula al día?',
            [0, 1],
            format_func=lambda x: 'No' if x == 0 else 'Sí',
            help="¿Está al día con el pago de matrícula?"
        )
        
        scholarship = st.selectbox(
            '¿Becario?',
            [0, 1],
            format_func=lambda x: 'No' if x == 0 else 'Sí',
            help="¿Cuenta con beca o ayuda económica?"
        )

    with col4:
        mother_qualification = st.selectbox(
            'Educación de la Madre',
            categories['mother_qualification'],
            help="Nivel educativo de la madre"
        )
        
        mother_occupation = st.selectbox(
            'Ocupación de la Madre',
            categories['mother_occupation'],
            help="Ocupación laboral de la madre"
        )

    with col5:
        father_qualification = st.selectbox(
            'Educación del Padre',
            categories['father_qualification'],
            help="Nivel educativo del padre"
        )
        
        father_occupation = st.selectbox(
            'Ocupación del Padre',
            categories['father_occupation'],
            help="Ocupación laboral del padre"
        )

    # Información académica detallada
    st.subheader("📚 Rendimiento Académico")
    col6, col7 = st.columns(2)

    with col6:
        st.markdown("**Calificaciones y Orden**")
        
        application_order = st.number_input(
            'Orden de Aplicación',
            min_value=0,
            max_value=100,
            value=1,
            help="Orden de preferencia de la carrera"
        )
        
        previous_qualification_grade = st.number_input(
            'Calificación Previa',
            min_value=0.0,
            max_value=200.0,
            value=120.0,
            step=0.1,
            help="Calificación de educación previa"
        )
        
        admission_grade = st.number_input(
            'Calificación de Admisión',
            min_value=0.0,
            max_value=200.0,
            value=120.0,
            step=0.1,
            help="Calificación obtenida en el examen de admisión"
        )

    with col7:
        st.markdown("**Primer Semestre**")
        
        cu_1st_evaluations = st.number_input(
            'Evaluaciones 1er Semestre',
            min_value=0,
            max_value=50,
            value=10,
            help="Número de evaluaciones en el primer semestre"
        )
        
        cu_1st_without_eval = st.number_input(
            'Sin Evaluaciones 1er Semestre',
            min_value=0,
            max_value=50,
            value=0,
            help="Unidades curriculares sin evaluación en el primer semestre"
        )

    # Segundo semestre
    st.markdown("**Segundo Semestre**")
    col8, col9, col10 = st.columns(3)

    with col8:
        cu_2nd_credited = st.number_input(
            'Creditadas 2do Sem',
            min_value=0,
            max_value=50,
            value=0,
            help="Unidades curriculares creditadas"
        )
        
        cu_2nd_enrolled = st.number_input(
            'Inscritas 2do Sem',
            min_value=0,
            max_value=50,
            value=10,
            help="Unidades curriculares inscritas"
        )

    with col9:
        cu_2nd_evaluations = st.number_input(
            'Evaluaciones 2do Sem',
            min_value=0,
            max_value=50,
            value=10,
            help="Número de evaluaciones"
        )
        
        cu_2nd_approved = st.number_input(
            'Aprobadas 2do Sem',
            min_value=0,
            max_value=50,
            value=8,
            help="Unidades curriculares aprobadas"
        )

    with col10:
        cu_2nd_grade = st.number_input(
            'Calificación 2do Sem',
            min_value=0.0,
            max_value=20.0,
            value=12.0,
            step=0.1,
            help="Calificación promedio del segundo semestre"
        )
        
        cu_2nd_without_eval = st.number_input(
            'Sin Evaluaciones 2do Sem',
            min_value=0,
            max_value=50,
            value=0,
            help="Unidades curriculares sin evaluación"
        )

    # Información macroeconómica
    st.subheader("📈 Indicadores Macroeconómicos")
    col11, col12, col13 = st.columns(3)

    with col11:
        unemployment_rate = st.number_input(
            'Tasa de Desempleo (%)',
            min_value=0.0,
            max_value=30.0,
            value=10.2,
            step=0.1,
            help="Tasa de desempleo en el momento de inscripción"
        )

    with col12:
        inflation_rate = st.number_input(
            'Tasa de Inflación (%)',
            min_value=-5.0,
            max_value=20.0,
            value=1.4,
            step=0.1,
            help="Tasa de inflación en el momento de inscripción"
        )

    with col13:
        gdp = st.number_input(
            'PIB',
            min_value=-10.0,
            max_value=10.0,
            value=1.74,
            step=0.01,
            help="Producto Interno Bruto en el momento de inscripción"
        )

    # Botón de predicción
    if st.button('🔮 Realizar Predicción', type="primary"):
        try:
            # Recopilar todos los datos
            user_data = {
                'application_order': application_order,
                'daytime_evening': daytime_evening,
                'previous_qualification_grade': previous_qualification_grade,
                'admission_grade': admission_grade,
                'displaced': displaced,
                'debtor': debtor,
                'tuition_fees': tuition_fees,
                'gender': gender,
                'scholarship': scholarship,
                'age_enrollment': age_enrollment,
                'cu_1st_evaluations': cu_1st_evaluations,
                'cu_1st_without_eval': cu_1st_without_eval,
                'cu_2nd_credited': cu_2nd_credited,
                'cu_2nd_enrolled': cu_2nd_enrolled,
                'cu_2nd_evaluations': cu_2nd_evaluations,
                'cu_2nd_approved': cu_2nd_approved,
                'cu_2nd_grade': cu_2nd_grade,
                'cu_2nd_without_eval': cu_2nd_without_eval,
                'unemployment_rate': unemployment_rate,
                'inflation_rate': inflation_rate,
                'gdp': gdp,
                'marital_status': marital_status,
                'application_mode': application_mode,
                'course': course,
                'previous_qualification': previous_qualification,
                'nationality': nationality,
                'mother_qualification': mother_qualification,
                'father_qualification': father_qualification,
                'mother_occupation': mother_occupation,
                'father_occupation': father_occupation
            }

            # Crear DataFrame completo
            df_entrada = create_complete_dataframe(user_data, categories)

            # Mostrar datos ingresados (resumen)
            st.subheader("📋 Resumen de Datos Ingresados")
            
            # Crear resumen más legible
            resumen = pd.DataFrame([
                ['Edad', f"{age_enrollment} años"],
                ['Género', 'Masculino' if gender == 1 else 'Femenino'],
                ['Estado Civil', marital_status],
                ['Programa', course],
                ['Modo de Aplicación', application_mode],
                ['Calificación de Admisión', f"{admission_grade:.1f}"],
                ['Calificación 2do Semestre', f"{cu_2nd_grade:.1f}"],
                ['Aprobadas 2do Sem', f"{cu_2nd_approved}"],
                ['¿Becario?', 'Sí' if scholarship == 1 else 'No'],
                ['¿Matrícula al día?', 'Sí' if tuition_fees == 1 else 'No']
            ], columns=['Variable', 'Valor'])
            
            st.dataframe(resumen, use_container_width=True)

            # Realizar predicción
            prediccion = pipeline.predict(df_entrada)[0]
            probabilidades = pipeline.predict_proba(df_entrada)[0]

            # Mostrar resultados
            st.header("🎯 Resultados de la Predicción")

            col14, col15 = st.columns(2)

            with col14:
                if prediccion == 1:
                    st.error("### ⚠️ RIESGO DE DESERCIÓN")
                    probabilidad_desercion = probabilidades[1]
                    st.error(f"**Probabilidad de Deserción:** {probabilidad_desercion:.1%}")
                    st.progress(probabilidad_desercion)
                    
                    # Recomendaciones
                    st.markdown("### 📝 Recomendaciones")
                    st.warning("""
                    - Brindar acompañamiento académico personalizado
                    - Ofrecer tutorías adicionales
                    - Evaluar opciones de apoyo económico
                    - Seguimiento psicológico y motivacional
                    - Revisión del plan de estudios personalizado
                    """)
                else:
                    st.success("### ✅ BAJO RIESGO DE DESERCIÓN")
                    probabilidad_graduacion = probabilidades[0]
                    st.success(f"**Probabilidad de Graduación:** {probabilidad_graduacion:.1%}")
                    st.progress(probabilidad_graduacion)
                    
                    st.markdown("### 📝 Recomendaciones")
                    st.info("""
                    - Mantener el buen rendimiento académico
                    - Participar en actividades extracurriculares
                    - Considerar programas de liderazgo estudiantil
                    - Explorar oportunidades de investigación
                    - Preparación para el mundo laboral
                    """)

            with col15:
                st.subheader("📊 Distribución de Probabilidades")
                
                # Crear gráfico de barras con Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Graduación', 'Deserción'],
                        y=[probabilidades[0], probabilidades[1]],
                        marker_color=['green', 'red'],
                        text=[f'{probabilidades[0]:.1%}', f'{probabilidades[1]:.1%}'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Probabilidades de Predicción",
                    yaxis_title="Probabilidad",
                    xaxis_title="Resultado",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Métricas detalladas
            st.subheader("📈 Análisis Detallado")
            
            col16, col17 = st.columns(2)
            with col16:
                st.metric(
                    "Probabilidad de Graduación", 
                    f"{probabilidades[0]:.4f}", 
                    f"{probabilidades[0]:.1%}"
                )
            with col17:
                st.metric(
                    "Probabilidad de Deserción", 
                    f"{probabilidades[1]:.4f}", 
                    f"{probabilidades[1]:.1%}"
                )

            # Información contextual
            st.subheader("📚 Información del Contexto")
            st.info(f"""
            **Interpretación de Resultados:**
            
            - **Graduación (0):** El estudiante tiene alta probabilidad de completar sus estudios exitosamente
            - **Deserción (1):** El estudiante tiene riesgo de abandonar los estudios antes de graduarse
            
            **Distribución en el conjunto de entrenamiento:**
            - Estudiantes que se graduaron: ~67%
            - Estudiantes que desertaron: ~33%
            
            **Confianza del Modelo:** F1-Score de 91% (Excelente performance)
            
            Esta predicción puede ayudar a las instituciones educativas a implementar 
            estrategias de retención estudiantil de manera proactiva y personalizada.
            """)

            # Factores de riesgo identificados
            if prediccion == 1:
                st.subheader("🚨 Factores de Riesgo Identificados")
                
                factores_riesgo = []
                
                if cu_2nd_grade < 10:
                    factores_riesgo.append("Calificación baja en segundo semestre")
                if cu_2nd_approved < cu_2nd_enrolled * 0.7:
                    factores_riesgo.append("Bajo porcentaje de materias aprobadas")
                if debtor == 1:
                    factores_riesgo.append("Situación de deuda")
                if tuition_fees == 0:
                    factores_riesgo.append("Matrícula no actualizada")
                if scholarship == 0 and debtor == 1:
                    factores_riesgo.append("Sin apoyo económico y con deudas")
                if age_enrollment > 25:
                    factores_riesgo.append("Ingreso tardío a la educación superior")
                
                if factores_riesgo:
                    for factor in factores_riesgo:
                        st.warning(f"• {factor}")
                else:
                    st.info("No se identificaron factores de riesgo específicos obvios.")

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.info("Verifique que todos los campos estén correctamente completados.")

else:
    st.error("❌ No se pudo cargar el pipeline del modelo.")
    st.info("📁 Asegúrese de que el archivo 'pipeline_final_desercion.pkl' esté en el directorio.")

# Footer
st.markdown("---")
st.markdown("**🎓 Sistema de Predicción de Deserción Estudiantil**")
st.markdown("*Modelo XGBoost optimizado con Bayesian Optimization | F1-Score: 91%*")
st.markdown("*Desarrollado siguiendo metodología CRISP-DM para Analítica de Datos*")