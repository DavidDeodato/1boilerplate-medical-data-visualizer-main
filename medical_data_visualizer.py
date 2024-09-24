import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar os dados
df = pd.read_csv('medical_examination.csv')

# 2. Adicionar coluna 'overweight'
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalizar os dados
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 5. Criar DataFrame para o gráfico categórico
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6. Agrupar e reformatar os dados
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()).rename(columns={'value': 'total'}).reset_index()
    
    # 7. Criar o gráfico categórico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    fig.set_axis_labels("variable", "total")
    fig.set_titles("Cardio - {col_name}")
    
    # 8. Obter o valor da saída
    fig = fig.fig

    # 9. Não modificar as próximas duas linhas
    fig.savefig('catplot.png')
    return fig

# 10. Função para desenhar o mapa de calor
def draw_heat_map():
    # 11. Limpar os dados
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcular a matriz de correlação
    corr = df_heat.corr()

    # 13. Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15. Plotar o mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', square=True, linewidths=.5, ax=ax, cbar_kws={"shrink": .5})

    # 16. Não modificar as próximas duas linhas
    fig.savefig('heatmap.png')
    return fig