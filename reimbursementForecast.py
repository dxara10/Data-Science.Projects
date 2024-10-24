# Importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carregue seus dados de histórico de reembolso em um DataFrame
data = pd.read_csv('dados_de_reembolso.csv')

# Divida os dados em conjuntos de treinamento e teste
X = data[['Tipo_de_Produto', 'Motivo_do_Reembolso', 'Historico_do_Cliente']]
y = data['Valor_do_Reembolso']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Faça previsões com o modelo
y_pred = model.predict(X_test)
