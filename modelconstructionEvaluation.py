from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def dividir_dados_treino_teste(X, y, tamanho_teste=0.2, estado_aleatorio=None):
    return train_test_split(X, y, test_size=tamanho_teste, random_state=estado_aleatorio)

def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    return {
        'acuracia': accuracy_score(y_teste, y_pred),
        'relatorio_classificacao': classification_report(y_teste, y_pred)
    }
