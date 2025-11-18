import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

sns.set(style="whitegrid")

df = pd.read_csv("Arquivos/salary.csv")

print("Valores únicos na coluna salary:", df['salary'].unique())
sns.countplot(x='salary', data=df)
plt.title("Distribuição do salário")
plt.show()

df['salary_class'] = df['salary'].str.strip().map({'<=50K':0, '>50K':1})
print("\nDistribuição das classes:\n", df['salary_class'].value_counts())

categorical_cols = df.select_dtypes(include=['object']).columns.drop('salary')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(["salary", "salary_class"], axis=1)
y = df_encoded["salary_class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ========================
# 8. Regressão Logística
# ========================
logreg = LogisticRegression(max_iter=5000, class_weight='balanced')
logreg.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)
y_prob_log = logreg.predict_proba(X_test)[:,1]

print("\n=== Regressão Logística ===")
print("Acurácia:", accuracy_score(y_test, y_pred_log))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coef": logreg.coef_[0]
})
coef_df["Odds_Ratio"] = np.exp(coef_df["Coef"])
print("\nCoeficientes e Odds Ratio:")
print(coef_df.sort_values(by="Odds_Ratio", ascending=False))

plt.figure(figsize=(8,4))
sns.histplot(y_prob_log, bins=20, kde=True)
plt.title("Distribuição das probabilidades preditas (Logística)")
plt.xlabel("Probabilidade de salário alto")
plt.ylabel("Frequência")
plt.show()

# ===============
# 9. Adicional
# ===============
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)
y_pred_lin_class = (y_pred_lin >= 0.5).astype(int)
print("\n=== Regressão Linear como Classificação ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lin_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lin_class))
print("Classification Report:\n", classification_report(y_test, y_pred_lin_class))