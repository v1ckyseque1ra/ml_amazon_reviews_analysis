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

## Modelos Probados y Resultados

Enumera los modelos de aprendizaje automático probados y los resultados obtenidos:

* **Modelo 1**: Breve descripción del modelo y sus hiperparámetros.
    * **Métricas de Evaluación**: Indica las métricas utilizadas para evaluar el rendimiento del modelo (ej. precisión, recall, F1-score, etc.).
    * **Resultados**: Presenta los resultados obtenidos para cada métrica.
* **Modelo 2**: (Repite el proceso para cada modelo probado).
* **Comparación de Modelos**: Compara los resultados de los diferentes modelos y selecciona el mejor modelo.

Ejemplo:

> Se probaron los siguientes modelos:
> * Regresión Logística:
>     * Métricas: Precisión, Recall, F1-score.
>     * Resultados: Precisión: 0.85, Recall: 0.78, F1-score: 0.81.
> * Árbol de Decisión:
>     * Métricas: Precisión, Recall, F1-score.
>     * Resultados: Precisión: 0.88, Recall: 0.82, F1-score: 0.85.
> * Random Forest:
>     * Métricas: Precisión, Recall, F1-score.
>     * Resultados: Precisión: 0.90, Recall: 0.85, F1-score: 0.87.
> > El modelo Random Forest obtuvo los mejores resultados en todas las métricas.

## Conclusiones y Aplicaciones Prácticas

Resume las principales conclusiones del proyecto y sus aplicaciones prácticas:

* **Conclusiones**: In process.
* **Aplicaciones Prácticas**:In process.
* **Trabajo Futuro**: Ampliacion del EDA. Optimización de hiperparámetros. Ampliación de la muestra. Debug y corrección de errores


## Instrucciones de Uso

Orden de ejecución de los notebooks:
Raw Data → EDA → Preprocesamiento → Modelado → Evaluación → Visualización  

Requisitos: pip install -r requirements.txt
## Librerías Utilizadas

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