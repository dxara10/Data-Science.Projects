# Importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Carregue seus dados de vendas em um DataFrame
data = pd.read_csv('dados_de_vendas.csv')

# Divida os dados em conjuntos de treinamento e teste
X = data[['Preco', 'Promocao', 'Sazonalidade']]
y = data['Vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Faça previsões com o modelo
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Crie visualizações dos resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Vendas Reais')
plt.ylabel('Previsões de Vendas')
plt.title('Desempenho do Modelo de Previsão de Vendas')
plt.show()

# Imprima métricas de desempenho
print(f'Erro Quadrático Médio: {mse}')
print(f'Coeficiente de Determinação (R²): {r2}')
