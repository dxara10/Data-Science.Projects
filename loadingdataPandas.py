import pandas as pd

def carregar_dados_csv(nome_arquivo):
    return pd.read_csv(nome_arquivo)

def carregar_dados_excel(nome_arquivo):
    return pd.read_excel(nome_arquivo)
