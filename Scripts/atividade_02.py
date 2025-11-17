import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Arquivos/data.csv')
print('DADOS:')
print(data.head(), '\n')

X = data[['glucose', 'bloodpressure']]
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = GaussianNB()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)

print(f"PREVISÃO: \n\n{y_pred}\n")
print("RELATÓRIO:")
print(classification_report( y_test, y_pred),'\n')
print(f'ACURÁCIA: {acuracia * 100:.2f}%')