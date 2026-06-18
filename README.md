# 📊 Predicción de Churn Bancario con Machine Learning Avanzado

Este proyecto implementa un pipeline robusto y profesional de Machine Learning de extremo a extremo para predecir la fuga de clientes (*churn*) en una entidad financiera. El enfoque principal no fue solo maximizar métricas teóricas, sino **optimizar el modelo en función del impacto financiero y las necesidades reales del negocio**.

---

## ☁️ Entorno de Desarrollo

Todo el proyecto ha sido desarrollado y ejecutado utilizando la infraestructura de **Google Colab**. Gracias a esto, el ecosistema de ejecución está optimizado para la nube, facilitando la reproducibilidad del pipeline y la instalación automatizada de dependencias complejas (como `catboost`) directamente en el entorno virtual.

---

## 🛠️ Estructura del Pipeline Paso a Paso

El desarrollo en el notebook de Colab se estructuró de forma modular para garantizar el rigor metodológico:

1. **Filtrado por Baja Varianza (`VarianceThreshold`):** Eliminación de variables numéricas con varianza inferior a 0.01 para limpiar el ruido inicial del dataset.
2. **Análisis de Multicolinealidad (Mapa de Calor):** Inspección visual mediante una matriz de correlación de Spearman. Se detectaron y eliminaron variables redundantes:
   * Se eliminó `age_bracket_encoded` en favor de `age` (para mantener la granularidad continua).
   * Se eliminó `is_zero_balance` debido a su alta correlación (0.85) con `balance` y `balance_to_salary_ratio`.
3. **Split Técnico Estratificado:** División del dataset en un 70% para entrenamiento y 30% para evaluación, aplicando `stratify` para preservar la proporción original de la variable objetivo (`churn`).
4. **Estandarización Selectiva (`StandardScaler`):** Se escalaron estrictamente las variables continuas, respetando el formato nativo (0/1) de las variables binarias y las procedentes de *One-Hot Encoding*.
5. **Balanceo de Clases (SMOTE):** Sobremuestreo sintético aplicado exclusivamente en el conjunto de entrenamiento, logrando un balance perfecto de 5,574 clientes estables vs. 5,574 en fuga.
6. **Competición de Modelos en Masa:** Se evaluaron 6 algoritmos bajo las mismas condiciones dentro del entorno de Colab: *Perceptron, SGDClassifier, LogisticRegression, GradientBoostingClassifier, XGBClassifier y CatBoostClassifier*.

---

## 📈 Resultados de la Competición

Los modelos basados en árboles dominaron la competición en su configuración por defecto (ordenados por su $F1$-Score):

| Modelo | Accuracy (Acierto General) | Recall (Captura de Churn) | F1-Score (Métrica Reina) |
| :--- | :---: | :---: | :---: |
| **CatBoostClassifier** | **86.23%** | 57.12% | **62.83%** |
| **GradientBoostingClassifier** | 83.47% | 68.09% | 62.65% |
| **XGBClassifier** | 85.53% | 57.12% | 61.66% |
| **LogisticRegression** | 72.50% | 68.41% | 50.33% |
| **SGDClassifier** | 67.20% | **76.27%** | 48.64% |
| **Perceptron** | 70.60% | 43.37% | 37.54% |

---

## 🎯 Optimización de Negocio: Ajuste de Umbral (Threshold Tuning)

Aunque **CatBoost** obtuvo el $F1$-Score más alto por defecto, un **Recall del 57% al 68% implicaba dejar escapar entre el 32% y el 43% de las fugas reales**, algo inaceptable para el negocio.

Tomando el **GradientBoostingClassifier**, se modificó el umbral de decisión por defecto (0.50) y se bajó de forma estratégica a **0.35 (35%)**, priorizando la sensibilidad del modelo para detectar clientes en riesgo.

### Impacto del Nuevo Umbral (35%):
* **Nuevo Recall:** **82.65%** (¡Se incrementó drásticamente la captura de fugas!).
* **Nuevo Accuracy:** 76.67%.

### Matriz de Confusión Resultante (Set de Test: 3,000 clientes):
* **Verdaderos Negativos (Fieles detectados):** 1,795
* **Verdaderos Positivos (Fugas interceptadas):** 505
* **Falsos Positivos (Falsas alarmas de riesgo):** 594
* **Falsos Negativos (Fugas no detectadas):** 106

**Justificación de Negocio:** En el sector bancario, el coste de lanzar una campaña de fidelización preventiva a un falso positivo (un cliente que no iba a irse) es infinitamente menor que el coste de perder un cliente real y tener que adquirir uno nuevo en el mercado. Con este ajuste, pasamos de perder cientos de clientes a dejar escapar **únicamente 106 de cada 3,000**.
