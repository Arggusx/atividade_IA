import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

matplotlib.use('Agg')

df = pd.read_csv("Arquivos/salary.csv")
print(df.head())

# Análise exploratória
df.info()
df.describe()

df['salary'].hist(bins=20)
plt.savefig("linear_hist_salary.png")
plt.close()

sns.pairplot(df)
plt.savefig("linear_pairplot.png")
plt.close()

print("Gráficos gerados!")

# ===========
# TRATAMENTO
# ===========

# Limpar espaços em branco nas colunas e nos valores (string)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Codificar a variável alvo (salary) para binário
# <=50K -> 0 e >50K -> 1
le = LabelEncoder()
df['salary_encoded'] = le.fit_transform(df['salary'])

# Selecionar features e alvo
# Removemos a coluna original de texto 'salary' e a nova codificada 'salary_encoded' das features
X = df.drop(['salary', 'salary_encoded'], axis=1)
y = df['salary_encoded']

# Converter variáveis categóricas restantes em numéricas (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

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
print(coef)
