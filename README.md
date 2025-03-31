# Heart Disease Prediction with the UCI Dataset
## Implementing Machine Learning Approaches

Project Brief: In this project, I use the UCI Heart Disease dataset to predict whether a patient has heart disease based on clinical features. I explore the data, handle missing values, and build logistic regression models. This project demonstrates skills in data wrangling, EDA, model evaluation, and feature engineering using Python.

### Data
The dataset is imported using the `ucimlrepo` library and includes demographic and clinical information. It originally has 5 target classes, which I recoded into a binary outcome: 0 = no disease, 1 = presence of disease.

```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
```

### Exploratory Data Analysis
- Count plots reveal a fairly balanced target class
- Handled 6 missing values using **mode imputation**
- Visualized relationships with `sns.pairplot` and a **correlation heatmap**
- Key insight: `slope` and `oldpeak` were highly correlated

### Target Encoding
Converted the multiclass target to binary

### Baseline Logistic Regression Model
- Split data (75/25)
- Trained using `LogisticRegression`
- Evaluated with accuracy, confusion matrix, and classification report

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_new, test_size=0.25, random_state=41)
lr = LogisticRegression(max_iter=1000, random_state=41)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

#### Baseline Results:
- **Accuracy**: 88%
- **Precision**: 88%
- **Recall**: 85%

### Feature Engineering & Comparisons

#### MinMax Scaling
```python
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X_imputed)
```
- Accuracy dropped to **80%**
- Possible reason: MinMax scaling may have negatively affected categorical features

#### SelectKBest
```python
from sklearn.feature_selection import SelectKBest, f_classif
X_k_best = SelectKBest(f_classif, k=5).fit_transform(X_imputed, y_new)
```
- Accuracy slightly improved to ~83–85%

### Model Comparison Table
| Model               | Accuracy | Notes                         |
|--------------------|----------|-------------------------------|
| Baseline Logistic  | 88%      | Highest performance           |
| MinMax Scaled      | 80%      | Lower; likely due to mixed feature types |
| SelectKBest        | 83–85%   | Better than scaling alone     |

### Conclusion
Logistic regression performs well in predicting heart disease with minimal preprocessing. Feature engineering did not outperform the baseline model but opened paths for future exploration such as encoding, tree-based models, and deeper variable selection.

This project reinforced the power of simplicity, strong baselines, and thoughtful analysis.

---
*See the full notebook for detailed code and visualizations.*
