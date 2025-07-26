# 📉 Telecom_Churn_Classification_Model__Supervised_Machine_Learning

This project focuses on predicting customer churn in a telecommunications company using supervised machine learning. Several classic classification models were implemented and evaluated under different class imbalance scenarios, including the application of the SMOTE oversampling technique.

The work was developed as part of the **Data Science, Machine Learning, Artificial Intelligence, and Deep Learning Diploma** at the Pontificia Universidad Católica de Valparaíso (PUCV), Chile, in 2025.

---

## 🧪 Techniques Used

### 🔍 Exploratory Data Analysis (EDA)
- Statistical summaries (`mean`, `std`, `min`, `max`, etc.)
- Visualizations:
  - Histograms and boxplots for numerical variables
  - Bar plots for categorical distributions
- Contingency tables to analyze relationships (e.g., international plan vs. churn)
- Outlier detection using the **interquartile range (IQR)** method

### ⚙️ Data Preprocessing
- Removal of non-informative variables (`phone number`, `customer ID`)
- Label encoding for binary variables (e.g., `International Plan`)
- One-hot encoding for categorical variables (e.g., `State`)
- Feature scaling using `StandardScaler`

### 🛠 Feature Engineering
- Derived feature: `customer_service_cat` (based on number of service calls)
  - Groups: 0–1 calls, 2–3 calls, 4 or more
- Detection of key churn predictors based on initial insights

### 🧠 Classification Models
- **Random Forest Classifier**
- **Support Vector Machine (SVM)** with RBF kernel

### ⚖️ Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied in two scenarios:
  1. SMOTE applied to both models
  2. Hybrid: RF without SMOTE, SVM with SMOTE

### 📏 Model Evaluation
- Metrics:
  - Accuracy
  - Precision, Recall, F1-Score (focused on churn class)
  - Area Under the Curve (AUC-ROC)
  - Precision-Recall Curve
- Confusion Matrices
- Visual comparison of performance across all models

---

## 📊 Key Results

| Model                                | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | SMOTE |
|-------------------------------------|----------|-------------------|----------------|------------------|--------|
| Random Forest (no SMOTE)            | 95%      | 0.92              | 0.70           | 0.80             | ❌     |
| SVM (no SMOTE)                      | 92%      | 0.88              | 0.50           | 0.63             | ❌     |
| Random Forest (SMOTE)               | 92%      | 0.70              | 0.74           | 0.72             | ✅     |
| SVM (SMOTE)                         | 87%      | 0.54              | 0.66           | 0.59             | ✅     |
| RF (no SMOTE) + SVM (with SMOTE)    | 95% / 87%| 0.92 / 0.54       | 0.70 / 0.66    | 0.80 / 0.59      | Mixed  |

> ✅ The best overall performance was achieved using **Random Forest without SMOTE**, reaching **95% accuracy** and an **F1-Score of 0.80** for the minority class (churn).

---

## 📂 Repository Structure
modelo-clasificacion-churn-telecom/
├── README.md
├── Gustavo_Alcántara_Modelo_de_Clasificación.ipynb # Full model in Jupyter Notebook
├── Gustavo_Alcántara_Modelo_de_Clasificación_Diplomado_IA_PUCV_Mayo_2025.pdf # Full report
├── data/
│ └── churn-analysis.csv # Dataset (if shareable)
└── images/
└── ... # Visualizations (optional)


---

## 🧰 Tools & Libraries

- Python 3.10+
- Jupyter Notebook
- `pandas`, `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`, `seaborn`



## 📈 Dataset
- The dataset consists of 3,333 customer records and 20 features, including:

- Call duration and charges (day, evening, night, international)

- Number of service calls

- Subscription to international plan and voicemail

- Target variable: churn (1 = customer left, 0 = customer stayed)

- Class imbalance: 14.5% churn rate

## 📌 Business Recommendations
Prioritize Random Forest without SMOTE for deployment due to its balance of performance and simplicity.

- Focus retention efforts on:

- Customers with international plans

- Customers with 4+ service calls

- Customers without voicemail services



# 📉 Modelo de Clasificación de Fuga de Clientes (Churn) - Telecomunicaciones

Este proyecto aborda el problema de **predicción de la fuga de clientes (churn)** en una empresa de telecomunicaciones, utilizando técnicas de **aprendizaje automático supervisado**. Se implementaron modelos clásicos de clasificación bajo distintos escenarios de desbalance de clases, incluyendo la técnica de sobremuestreo **SMOTE**.

Este trabajo fue desarrollado en el contexto del **Diplomado en Ciencia de Datos, Machine Learning, Inteligencia Artificial y Deep Learning** de la **Pontificia Universidad Católica de Valparaíso (PUCV)**, Chile, en el año 2025.

---

## 🧪 Técnicas utilizadas

### 🔍 Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas (`media`, `desv. estándar`, `mínimo`, `máximo`, etc.)
- Visualizaciones:
  - Histogramas y boxplots para variables numéricas
  - Gráficos de barras para variables categóricas
- Tablas de contingencia para explorar relaciones entre variables
- Detección de valores atípicos utilizando el **rango intercuartílico (IQR)**

### ⚙️ Preprocesamiento de datos
- Eliminación de variables no informativas (`número de teléfono`, `ID cliente`)
- Codificación de variables:
  - `Label Encoding` para binarias
  - `One-Hot Encoding` para categóricas con múltiples valores
- Normalización de variables numéricas con `StandardScaler`

### 🛠 Ingeniería de características
- Variable derivada: `customer_service_cat` (categoriza número de llamadas)
  - Grupos: 0–1 llamadas, 2–3 llamadas, 4 o más
- Selección de atributos con mayor poder predictivo

### 🧠 Modelos de Clasificación
- **Random Forest Classifier**
- **Support Vector Machine (SVM)** con kernel RBF

### ⚖️ Manejo de desbalance de clases
- Aplicación de **SMOTE** en dos escenarios:
  1. SMOTE aplicado a ambos modelos
  2. Enfoque híbrido: Random Forest sin SMOTE, SVM con SMOTE

### 📏 Evaluación de modelos
- Métricas:
  - Accuracy
  - Precisión, Recall, F1-Score (especialmente para clase *churn*)
  - Curva ROC (AUC)
  - Curva Precision-Recall
- Matrices de confusión
- Comparación visual entre modelos

---

## 📊 Resultados clave

| Modelo                              | Accuracy | Precisión (Churn) | Recall (Churn) | F1-Score (Churn) | SMOTE |
|------------------------------------|----------|-------------------|----------------|------------------|--------|
| Random Forest (sin SMOTE)          | 95%      | 0.92              | 0.70           | 0.80             | ❌     |
| SVM (sin SMOTE)                    | 92%      | 0.88              | 0.50           | 0.63             | ❌     |
| Random Forest (con SMOTE)          | 92%      | 0.70              | 0.74           | 0.72             | ✅     |
| SVM (con SMOTE)                    | 87%      | 0.54              | 0.66           | 0.59             | ✅     |
| RF (sin SMOTE) + SVM (con SMOTE)   | 95% / 87%| 0.92 / 0.54       | 0.70 / 0.66    | 0.80 / 0.59      | Mixto  |

> ✅ El mejor desempeño general fue obtenido con **Random Forest sin SMOTE**, alcanzando un **95% de accuracy** y un **F1-Score de 0.80** para la clase minoritaria (*churn*).

---

## 📂 Estructura del repositorio


modelo-clasificacion-churn-telecom/
├── README.md
├── Gustavo_Alcántara_Modelo_de_Clasificación.ipynb # Notebook completo
├── Gustavo_Alcántara_Modelo_de_Clasificación_Diplomado_IA_PUCV_Mayo_2025.pdf
├── data/
  └── churn-analysis.csv

---

## 🧰 Herramientas y librerías

- Python 3.10+
- Jupyter Notebook
- `pandas`, `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`, `seaborn`

---

## 📈 Dataset
- El conjunto de datos contiene 3.333 registros de clientes y 20 atributos, incluyendo:

- Duración y cargos de llamadas (día, tarde, noche, internacional)

- Número de llamadas al servicio al cliente

- Planes contratados (internacional, buzón de voz)

- Variable objetivo: churn (1 = abandono, 0 = permanencia)

- Distribución: 14.5 % de churn (desbalance significativo)

---

## 📌 Recomendaciones para el negocio
- Adoptar el modelo de Random Forest sin SMOTE como herramienta base de predicción.

- Focalizar estrategias de retención en:

- Clientes con planes internacionales

- Clientes que realizaron 4 o más llamadas al servicio

- Clientes sin buzón de voz

---

## 🧑 Autor
Gustavo Alcántara Aravena

📘 Diplomado en Ciencia de Datos, Machine Learning, IA, Deep Learning
Pontificia Universidad Católica de Valparaíso (PUCV)
📍 Valparaíso, Chile | 2025
