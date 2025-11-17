import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use('Agg')

df = pd.read_csv("Arquivos/salary.csv")
print(df.head())

# Análise exploratória
df.info()
df.describe()

df['salary'].hist(bins=20)
plt.savefig("hist_salary.png")
plt.close()

sns.pairplot(df)
plt.savefig("pairplot.png")
plt.close()

print("Gráficos gerados!")

# ===========
# TRATAMENTO 
# ===========

# Separar variável-alvo ANTES do get_dummies
y = df["salary"]
y = (y == ">50K").astype(int)  # converter para binário

df = df.drop("salary", axis=1)

# Preencher nulos
df = df.fillna(df.median(numeric_only=True))

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df

# ==========
# TREINO
# ==========

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========================
# REGRESSÃO LINEAR
# ========================

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# MÉTRICAS
mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R²: ", r2)

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": lr.coef_
})
coef
