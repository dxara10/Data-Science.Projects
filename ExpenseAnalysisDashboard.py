import plotly.express as px

# Dados de exemplo (substitua por seus próprios dados)
categorias = ['Aluguel', 'Alimentação', 'Transporte', 'Lazer', 'Outros']
despesas = [1200, 500, 300, 200, 700]

# Crie um gráfico de barras interativo
fig = px.bar(x=categorias, y=despesas, labels={'x': 'Categorias', 'y': 'Despesas (em R$)'},
             title='Análise de Despesas por Categoria')

# Personalize o layout do gráfico
fig.update_layout(xaxis_title_font=dict(size=14, family='Arial'),
                  yaxis_title_font=dict(size=14, family='Arial'),
                  title_font=dict(size=16, family='Arial'),
                  xaxis=dict(categoryorder='total descending'))

# Exiba o gráfico interativo
fig.show()
