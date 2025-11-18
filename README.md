# Comparação entre modelos com regressão linear e logística (Atividade 3)

### Como executar

No diretório do projeto crie o ambiente virtual do Python (venv):

```
python -m venv ./ia-venv
```

Entre no ambiente virtual:

(Linux)
```
source ./ia-venv/bin/activate
```

(Windows)
```
.\ia-venv\Scripts\activate.bat
```

Instale as dependências nesse ambiente virtual:

```
python -m pip install pandas scikit-learn seaborn matplotlib
```

Para executar o script de regressão linear, execute:

```
python Scripts/Atividade_03/regresLinear.py
```

Para executar o script de regressão logística, execute:

```
python Scripts/Atividade_03/regresLogistica.py
```

### Processo de execução
- Carregamento e tratamento do dataset (remoção de caracteres inválidos, limpeza, encoding).

- Separação entre tipos de variáveis diferentes (numéricas e categóricas).

- Divisão do conjunto de dados em treino e teste.

- Aplicação dos modelos:
  - Regressão Linear (regressão).
  - Regressão Logística (classificação).

- Avaliação dos modelos com métricas apropriadas.

- Geração de gráficos (Salário (Hist), PairPlot, Distribuição das probabilidades preditas, Distribuição do salário).

### Resumo
O objetivo central é prever se um indivíduo recebe acima ou abaixo de 50 mil anuais. O trabalho iniciou-se com uma rigorosa preparação da base de dados, que abrangeu a limpeza de registros, o tratamento de valores ausentes, a codificação de variáveis categóricas e a padronização das numéricas, garantindo assim a qualidade necessária para a modelagem.

Na etapa de testes, comparamos o desempenho da Regressão Linear com a Regressão Logística. Os resultados evidenciaram que a Regressão Linear é inadequada para este cenário; por ser desenhada para estimar valores contínuos, ela não consegue lidar corretamente com a natureza binária da variável alvo, falha que foi confirmada pela análise dos resíduos e pela inconsistência das previsões.

Em contrapartida, a Regressão Logística demonstrou ser a abordagem ideal. O modelo não apenas apresentou um desempenho superior (validado por uma alta precisão, uma matriz de confusão sólida e uma Curva ROC/AUC eficaz), mas também ofereceu valiosa interpretabilidade através de seus coeficientes, permitindo entender claramente quais fatores aumentam ou diminuem a probabilidade de altos salários. Portanto, conclui-se que a Regressão Logística é a ferramenta recomendada para este problema, unindo precisão estatística à capacidade de explicar o fenômeno estudado.
