# Amazon Reviews Sentiment Analysis

## Problem Description

### Context
Sentiment analysis on Amazon reviews aims to automatically extract insights about customers' opinions toward products, brands, or services. This helps to:
- Measure customer satisfaction
- Identify recurring issues
- Compare competing products
- Optimize business decisions

### Relevance
Depending on stakeholder needs, this analysis can answer:
- For sellers: "Why did our sales/ratings drop?" or "What features should our new product have?"
- For buyers: "How can I evaluate product quality from thousands of reviews?"
- Primary focus: Detecting fake or manipulated reviews on Amazon platforms

### Objectives
- Eliminate time-consuming manual review reading
- Remove human analyst bias
- Detect reputation crises before they impact sales

## Data Processing Pipeline

### 1. Data Collection
- Source: McAuley Lab (2023) - [Amazon Reviews Dataset](https://amazon-reviews-2023.github.io/)
- Sampled category: All Beauty products

### 2. Data Cleaning
- Removed null values using `dropna()`
- Eliminated irrelevant columns (images, metadata)
- Removed duplicates with `drop_duplicates()`
- Preserved original data by creating `df_reviews_clean` copy

### 3. Data Transformation
- Column concatenation
- Text normalization (case folding)
- Stopword removal
- Tokenization (word-level)
- Sentiment labeling (positive/neutral/negative by rating)
- Exported processed data to CSV

### 4. Exploratory Data Analysis (EDA)
- Initial structural analysis
- Rating distribution visualization (bar plots)
- Verified purchase frequency analysis

## 🤖 Tested Models and Results

We evaluated multiple supervised classification models for predicting review ratings in Amazon's "All Beauty" category:

### 🔸 Model 1: Naïve Bayes (Multinomial)
- **Type**: Probabilistic (Bayes theorem)
- **Vectorization**: TF-IDF
- **Results**:
  - Accuracy: 61%
  - F1 Macro: 0.43
  - F1 Weighted: 0.61
  - Speed: ⚡ Very fast
- **Findings**:
  - Strong on 5-star reviews
  - Weak on 1-4 star classes

### 🔸 Model 2: Logistic Regression (GridSearchCV)
- **Type**: Linear classifier with hyperparameter tuning
- **Results**:
  - Accuracy: 67%
  - F1 Macro: 0.44
  - F1 Weighted: 0.62
  - Speed: ⏱️ Medium
- **Findings**:
  - Best overall performer
  - Struggled with mid-range ratings (2-4 stars)

[... Additional models follow same structure...]

## 📊 Model Comparison

| Model               | Accuracy | F1 Macro | F1 Weighted | Speed     | Notes                      |
|---------------------|----------|----------|-------------|-----------|----------------------------|
| Naïve Bayes         | 0.61     | 0.43     | 0.61        | ⚡ Fast    | Fast but weak on minorities|
| Logistic Regression | 0.67     | 0.44     | 0.62        | ⏱️ Medium | Best balanced performance  |
| SVM                 | 0.65     | 0.42     | 0.60        | 🐌 Slow   | Good precision-recall      |

## 🏆 Selected Model: Logistic Regression

**Rationale**: Best balance between overall accuracy, minority class performance, and computational efficiency.

## Key Findings and Applications

### Conclusions
1. Logistic regression showed optimal performance after hyperparameter tuning
2. Model excels at extreme ratings (1/5 stars) but struggles with mid-range ratings
3. Text preprocessing was crucial for model performance
4. Dataset imbalance (mostly 5-star reviews) affected results

### Practical Applications
1. Automated review filtering for e-commerce platforms
2. Early reputation crisis detection
3. Support for business decisions (product improvements)
4. Potential fake review detection (future work)

### Future Improvements
- Dataset rebalancing techniques
- Advanced embeddings (Word2Vec, BERT)
- Expanded category coverage
- Interactive visualization tools

## Usage Instructions

**Execution order**:
1. Raw Data
2. EDA
3. Preprocessing
4. Modeling
5. Evaluation
6. Visualization

**Requirements**: `pip install -r requirements.txt`

## Libraries Used

```python
# Core
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# [...] (other imports)