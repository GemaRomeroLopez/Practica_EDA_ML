# Practica_EDA_ML

#### **Análisis Exploratorio y Preparación de Datos**

El análisis tiene como objetivo explorar las características del conjunto de datos sobre solicitudes de préstamos y prepararlos para su uso en modelos predictivos. Este enfoque asegura un entendimiento profundo del dataset y su vínculo con la variable objetivo (probabilidad de incumplimiento de pago), estableciendo una base sólida para el modelado.

#### **Descripción del Dataset**
El dataset incluye características individuales de los solicitantes y garantiza la privacidad de los usuarios. Su análisis busca evaluar cómo estas variables influyen en la capacidad de pago, centrándose especialmente en la variable objetivo:

- Valor 1: Dificultades para pagar el préstamo.
- Valor 0: No hubo dificultades.

#### **Objetivos del Análisis**
- Familiarizarse con el dataset y sus características.  
- Examinar la distribución de las variables y sus relaciones, especialmente con la variable objetivo.  
- Preparar los datos para su uso en algoritmos de Machine Learning.   
- Identificar patrones, relaciones significativas y estructuras principales en el dataset.  

-----------------------------------------------------------------------------------------

### **Guía del Trabajo**
El análisis se organiza en tres notebooks, cada uno enfocado en etapas específicas:

#### **Notebook 1: First_Exploration**
Centrado en el análisis preliminar y la comprensión del dataset. Incluye:

1. Análisis inicial: Exploración del dataset para garantizar su correcta manipulación.
2. Variable objetivo: Evaluación de su distribución, balance y características.
3. Valores faltantes: Identificación y análisis de datos missing, su relación con la variable objetivo y decisiones para su manejo.
4. Variables independientes: Exploración de su distribución, patrones y posibles influencias sobre la variable objetivo.

#### **Notebook 2: Data_Treatment**
Orientado al procesamiento de datos para optimizar el rendimiento del modelo. Incluye:

1. Separación Train/Test: División estratificada para evaluar la capacidad de generalización del modelo.
2. Tratamiento de outliers: Detección y manejo de valores atípicos que podrían afectar los resultados del modelo.
3. Relaciones entre variables:
- Matriz de correlación (Pearson): Identifica multicolinealidad entre variables numéricas.
- Matriz de Cramer (V de Cramer): Evalúa asociaciones entre variables categóricas y su relación 4. con la variable objetivo.
WOE e Information Value: Herramientas para evaluar la capacidad predictiva de las variables.

#### **Notebook 3: Data_Encoding_Scaling**
Enfocado en transformar los datos para su compatibilidad con algoritmos de machine learning. Incluye:

1. Codificación de variables categóricas:
Métodos como One-Hot Encoding, Target Encoding o CatBoost Encoding, según el tipo y características de las variables.
2. Escalamiento de variables numéricas:
Normalización o estandarización para garantizar una escala uniforme y mejorar el rendimiento del modelo.

-----------------------------------------------------------------------------------------
### **Estructura del repositorio**

- notebooks/: Archivos Jupyter con cada etapa del análisis.
- html/: Archivos HTML generados durante el análisis.
- src/: Funciones auxiliares en Python.
