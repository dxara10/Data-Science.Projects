import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregue os dados históricos de demanda de energia
data = pd.read_csv('dados_de_energia.csv')

# Separe as características (features) e o alvo (target)
X = data[['temperatura', 'umidade', 'hora']]
y = data['demanda']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crie um modelo de regressão linear
model = LinearRegression()

# Treine o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Faça previsões com os dados de teste
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro médio quadrático: {mse}")
