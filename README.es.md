# Amazon Reviews Sentiment Analysis

## Descripción del Problema

* **Contexto**: El análisis de sentimiento en reseñas de Amazon busca extraer insights automáticamente sobre la opinión de los clientes hacia productos, marcas o servicios. Esto ayuda a:
Medir la satisfacción del cliente.
Identificar problemas recurrentes.
Comparar productos competidores.
Optimizar decisiones de negocio.

* **Relevancia**: Dependiendo quien tuviera la necesidad de resoverlo, puede responder a preguntas como:
Porque bajaron nuestras ventas o valoraciones? o "Queremos lanzar un neuvo producto pero no sabemos que pide el mercado", si se tratara de un vendedor. "Hay miles de resenas, como se si el producto es bueno? de tratarse de un comprador. El foco en este caso estara en el uso de plataformas de Amazoon para detectar reseñas falsas o manipuladas.

* **Objetivos**: Se espera evitar pérdidas de tiempo al no tener que leer manualmente miles de reseñas, evitar sesgos humanos que un analista podria tener y detectar crisis de reputacion antes de que afecten las ventas.

## Proceso de Procesamiento de Datos

Detalla los pasos seguidos para preparar y procesar los datos:

1.  **Recopilación de Datos**: el dataset fue recolectado por McAuley Lab en 2023 (https://amazon-reviews-2023.github.io/ ) y la categoria elegida para muestreo fue All_beauty
2.  **Limpieza de Datos**: Se utilizaron las técnicas de: eliminación de valores nulos (NaN) dropna() para eliminar filas con valores faltantes; eliminacion de columnas innecesarias (images, entre otras irrelevantes para el análisis); eliminación de duplicados con drop_duplicates() para eliminar filas repetidas y se creó una copia del DF original df_reviews_clean antes de modificar los datos, preservando el original para evitar pérdidas accidentales.
3.  **Transformación de Datos**: Las operaciones realizadas para preparar los datos para el análisis fueron: Concatenación de columnas; Normalización del texto eliminando distinciones entre mayúsculas/minúsculas; Eliminación de stopwords, para remover palabras irrelevantes; Tokenización, es decir dividir el texto en unidades como palabras; Clasificación de sentimientos, etiquetando reseñas como positivo, neutral o negativo según la columna 'rating'; finalmente se exportó el dataframe procesado en un archivo .csv.
4.  **Análisis Exploratorio de Datos (EDA)**: luego del análisis inicial de la estructura, se realizaron visualizaciones de la distribución de la columna 'ratings' con gráficos de barras para verificar si hay desbalance de categorías, así como también la frecuencia de las compras verificadas 'verified_purchase'

## 🤖 Modelos Probados y Resultados

Se probaron distintos modelos de clasificación supervisada para predecir la puntuación de reseñas en la categoría `All_beauty` del dataset de Amazon. A continuación se detallan los modelos evaluados, las métricas empleadas y los resultados obtenidos:

---

### 🔸 Modelo 1: Naïve Bayes (Multinomial)

- **Descripción**: Modelo probabilístico basado en la aplicación del teorema de Bayes con independencia entre características. Se aplicó una versión optimizada sobre los datos vectorizados con TF-IDF.
- **Métricas de Evaluación**: Accuracy, F1-score Macro y Weighted.
- **Resultados**:
  - Accuracy: **61%**
  - F1-Score Macro: **0.43**
  - F1-Score Weighted: **0.61**
  - Tiempo de ejecución: ⚡ **Muy rápido**
  - Observaciones:
    - Buen rendimiento en clase 5.
    - Bajo desempeño en clases 1, 2, 3 y 4.
    - Promedio macro bajo por mal desempeño en clases minoritarias.

---

### 🔸 Modelo 2: Regresión Logística (con GridSearchCV)

- **Descripción**: Clasificador lineal optimizado mediante búsqueda de hiperparámetros con validación cruzada.
- **Métricas de Evaluación**: Accuracy, F1-score Macro y Weighted.
- **Resultados**:
  - Accuracy: **67%**
  - F1-Score Macro: **0.44**
  - F1-Score Weighted: **0.62**
  - Tiempo de ejecución: ⏱️ **Medio**
  - Observaciones:
    - Excelente desempeño en clase 5.
    - Dificultades en clases 2, 3 y 4 (bajo recall).
    - Mejor balance general comparado con Naïve Bayes.

---

### 🔸 Modelo 3: SVM (Support Vector Machine)

- **Descripción**: Clasificador de márgenes máximos con kernel lineal, optimizado con búsqueda de hiperparámetros.
- **Métricas de Evaluación**: Accuracy, F1-score Macro y Weighted.
- **Resultados**:
  - Accuracy: **65%**
  - F1-Score Macro: **0.42**
  - F1-Score Weighted: **0.60**
  - Tiempo de ejecución: 🐌 **Lento**
  - Observaciones:
    - Alto recall en clase 5.
    - Dificultades similares a Regresión Logística en clases intermedias.
    - Mejor balance entre precisión y recall que Gradient Boosting.

---

### 🔸 Modelo 4: Gradient Boosting

- **Descripción**: Ensemble basado en árboles secuenciales, entrenados para corregir errores de los anteriores.
- **Métricas de Evaluación**: Accuracy, F1-score Macro y Weighted.
- **Resultados**:
  - Accuracy: **65%**
  - F1-Score Macro: **0.37**
  - F1-Score Weighted: **0.58**
  - Tiempo de ejecución: ⏱️ **Medio**
  - Observaciones:
    - No detecta la clase 2 (recall = 0%).
    - Confusión frecuente entre clase 3 y 5.
    - Overfitting en ciertas clases.
    - Recomendado ajustar `min_samples_leaf` y utilizar `class_weight='balanced'`.

---

### 🔸 Modelo 5: XGBoost (Optimizado)

- **Descripción**: Algoritmo de boosting con regularización y control avanzado del sobreajuste.
- **Métricas de Evaluación**: Accuracy, F1-score Macro y Weighted.
- **Resultados**:
  - Accuracy: **56%**
  - F1-Score Macro: **0.19**
  - F1-Score Weighted: **0.46**
  - Tiempo de ejecución: ⏱️ **Medio**
  - Observaciones:
    - Pésimo rendimiento en clases 1 a 4.
    - Alto recall en clase 5 (>90%), indicando sesgo hacia la clase mayoritaria.
    - Posible overfitting; se sugiere reducir `max_depth` y ajustar `scale_pos_weight`.

---

### 📊 Comparación de Modelos

| Modelo                  | Accuracy | F1 Macro | F1 Weighted | Velocidad | Notas Principales |
|-------------------------|----------|----------|-------------|-----------|-------------------|
| Naïve Bayes             | 0.61     | 0.43     | 0.61        | ⚡ Rápido  | Rápido pero débil con clases minoritarias |
| Regresión Logística     | 0.67     | 0.44     | 0.62        | ⏱️ Medio  | Mejor rendimiento general |
| SVM                     | 0.65     | 0.42     | 0.60        | 🐌 Lento   | Buen balance, pero costoso |
| Gradient Boosting       | 0.65     | 0.37     | 0.58        | ⏱️ Medio  | Problemas con clase 2 |
| XGBoost                 | 0.56     | 0.19     | 0.46        | ⏱️ Medio  | Overfitting severo |

---

### 🏆 Modelo Seleccionado: **Regresión Logística**

Debido a su buen equilibrio entre precisión general, rendimiento aceptable en clases minoritarias, y velocidad razonable de entrenamiento, se selecciona **Regresión Logística** como el mejor modelo para esta tarea. Se recomienda seguir trabajando en mejorar su recall en clases 2, 3 y 4 con técnicas de rebalanceo o embeddings más ricos semánticamente.

## Conclusiones y Aplicaciones Prácticas

### 🔹 Conclusiones Principales

1. **Regresión Logística optimizada fue el mejor modelo**: Tras realizar ajustes de hiperparámetros con GridSearchCV, la regresión logística mostró un rendimiento sólido, destacando en precisión general y desempeño equilibrado entre clases. Esto sugiere que este modelo es eficaz para capturar patrones en los datos textuales de reseñas, especialmente en la categoría `All_beauty`.

2. **Desempeño por clase**: El modelo tuvo un excelente rendimiento para identificar reseñas con puntuaciones extremas, como las de 5 estrellas (`f1-score = 0.79`) y aceptable en las de 1 estrella. Sin embargo, presenta limitaciones para clasificar correctamente las opiniones intermedias (2, 3 y 4 estrellas), reflejo de un desbalance en el dataset y la ambigüedad natural de reseñas neutrales.

3. **Importancia del preprocesamiento**: Las técnicas de limpieza, tokenización, eliminación de stopwords y vectorización TF-IDF fueron fundamentales para transformar texto en datos útiles para el modelo. Estas etapas marcaron la diferencia en la calidad de predicción.

4. **Dataset desequilibrado**: El análisis exploratorio evidenció un claro predominio de reseñas positivas (mayormente de 5 estrellas), lo que afectó la capacidad de los modelos para clasificar correctamente opiniones negativas o neutras. Esto debe tenerse en cuenta para futuras optimizaciones.

---

### 🎯 Aplicaciones Prácticas

1. **Filtrado automático de reseñas**: Este sistema puede implementarse en plataformas de e-commerce para clasificar de forma automática miles de reseñas, ahorrando tiempo a analistas y usuarios que desean entender rápidamente la opinión general sobre un producto.

2. **Detección temprana de crisis de reputación**: Al identificar aumentos en reseñas negativas o cambios en la distribución de puntuaciones, el modelo puede ser una herramienta clave para alertar sobre problemas de calidad, logística o atención al cliente antes de que impacten significativamente las ventas.

3. **Apoyo a decisiones comerciales**: Vendedores pueden usar este análisis para evaluar qué atributos valoran más los consumidores, detectar puntos débiles, y adaptar sus productos o estrategias de marketing en consecuencia.

4. **Evaluación de reseñas falsas o manipuladas** *(futuro trabajo)*: A futuro, se podría incorporar un modelo que identifique patrones atípicos en reseñas muy positivas o negativas, ayudando a Amazon a detectar comportamientos sospechosos y mejorar la confianza en la plataforma.

---

### 🚀 Trabajo Futuro

- **Rebalanceo del dataset** con técnicas como *undersampling*, *oversampling* o *class weighting* para mejorar la clasificación en clases minoritarias.
- **Entrenamiento con embeddings** tipo Word2Vec, GloVe o modelos BERT para capturar mejor la semántica del lenguaje.
- **Ampliar la muestra de reseñas** a otras categorías para validar la generalización del modelo.
- **Integración con herramientas de visualización interactiva** para monitoreo en tiempo real de la percepción del cliente.



## Instrucciones de Uso

Orden de ejecución de los notebooks:
Raw Data → EDA → Preprocesamiento → Modelado → Evaluación → Visualización  

Requisitos: pip install -r requirements.txt

# Librerías Utilizadas en el Proyecto

```python
import os
import re
import warnings
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bootcampviztools import pinta_distribucion_categoricas
from nltk.corpus import stopwords
import nltk
import json
import random

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import os
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import datetime
import pickle
import os
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


## Vicky Sequeira