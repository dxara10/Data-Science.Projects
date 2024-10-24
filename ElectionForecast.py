# Importe as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Carregue os dados de treinamento e teste (substitua pelo seu conjunto de dados)
dados_treinamento = pd.read_csv('dados_treinamento.csv')
dados_teste = pd.read_csv('dados_teste.csv')

# Divida os dados em recursos (features) e rótulos (labels)
X_treinamento = dados_treinamento.drop(columns=['resultado_eleição'])
y_treinamento = dados_treinamento['resultado_eleição']
X_teste = dados_teste.drop(columns=['resultado_eleição'])
y_teste = dados_teste['resultado_eleição']

# Crie e treine um modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X_treinamento, y_treinamento)

# Faça previsões com o modelo
previsoes = modelo.predict(X_teste)

# Avalie o desempenho do modelo
acuracia = accuracy_score(y_teste, previsoes)
relatorio_classificacao = classification_report(y_teste, previsoes)

print(f'Acurácia do modelo: {acuracia}')
print(f'Relatório de Classificação:\n{relatorio_classificacao}')
