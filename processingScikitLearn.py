from sklearn.preprocessing import StandardScaler, LabelEncoder

def padronizar_dados(dados):
    scaler = StandardScaler()
    return scaler.fit_transform(dados)

def codificar_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)
