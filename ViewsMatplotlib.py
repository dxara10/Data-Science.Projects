import matplotlib.pyplot as plt

def plotar_grafico(x, y, titulo='', xlabel='', ylabel=''):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
