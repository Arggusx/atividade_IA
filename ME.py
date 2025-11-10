import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

print(data.head())

# Separando as variáveis de entrada (temperatura e umidade) e o alvo (jogar_tenis)
X = data[['glucose', 'bloodpressure']]
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = GaussianNB()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)

print('Previsão: ', y_pred)

print(f'Acurácia: {acuracia * 100:.2f}%')