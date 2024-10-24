import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregue os dados históricos de cliques em anúncios
data = pd.read_csv('dados_de_anuncios.csv')

# Separe as características (features) e o alvo (target)
X = data[['palavras_chave', 'localizacao', 'dispositivo']]
y = data['cliques']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crie um modelo de classificação com Random Forest
model = RandomForestClassifier()

# Treine o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Faça previsões com os dados de teste
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")
