import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Use a biblioteca yfinance para obter dados de preços de ações (instale-a primeiro)
ticker = 'AAPL'  # Exemplo: ação da Apple
data = yf.download(ticker, start='2020-01-01', end='2021-01-01')

# Crie um DataFrame com os dados
df = data['Adj Close'].reset_index()

# Vamos ajustar um modelo de regressão linear simples
dias = np.arange(len(df)).reshape(-1, 1)
precos = df['Adj Close'].values.reshape(-1, 1)
modelo = LinearRegression()
modelo.fit(dias, precos)

# Preveja o preço para o próximo dia
dia_seguinte = np.array([[len(df)]])
previsao = modelo.predict(dia_seguinte)

print("Previsão do preço das ações para o próximo dia:", previsao[0][0])

# Vamos criar um gráfico para visualizar a tendência de preços
plt.plot(df['Date'], df['Adj Close'], label='Preço das Ações')
plt.plot(df['Date'].iloc[-1] + pd.Timedelta(days=1), previsao[0], 'ro', label='Previsão')
plt.xlabel('Data')
plt.ylabel('Preço das Ações')
plt.legend()
plt.show()
