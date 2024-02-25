import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("creditcard.csv")

# Carregar o DataFrame a partir do arquivo CSV

df_Fraude = df[df.Class == 1]
df_nao_fraude = df[df.Class == 0]
df_nao_fraude = df_nao_fraude.sample(492)
df_reg = pd.concat([df_nao_fraude, df_Fraude], axis = 0)
df_reg.reset_index(inplace = True)

#Dados validação do modelo
df_valNFraude = df_reg.head(10)
df_valFraude = df_reg.tail(10)

#Retitando linhas de validação
df_reg = df_reg.iloc[10:]
df_reg = df_reg.iloc[:-10]


df_validacao = pd.concat([df_valNFraude, df_valFraude])
df_validacao.reset_index(inplace = True)
df_validacao_Classe = df_validacao.Class
df_validacao = df_validacao.drop(['Time', 'Class', 'index', 'level_0'], axis = 1)

X = df_reg.drop(['index', 'Time', 'Class'], axis=1)
Y = df_reg['Class']

#Separação entre treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.35, random_state=18, stratify=Y)

#Treinamento
lr = LogisticRegression(max_iter=1500)
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
acc = accuracy_score(Y_test, pred)

#Validação
pred = lr.predict(df_validacao)
df_avaliacao = pd.DataFrame({'real': df_validacao_Classe, 'previsão': pred})

st.set_page_config(layout="wide")
st.title('Monitoramento TargeTech')
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

st.markdown('<h3 style="text-align: center;">Fraudes em Cartão de Crédito</h3>', unsafe_allow_html=True)

print("Classificador de Fraudes em cartões de crédito")
print("Acurácia do modelo: ", acc * 100)
perdas = st.sidebar.selectbox("Fraude", df_reg['Class'].unique())
df_filter = df_reg[df_reg['Class'] == perdas]
df_filter

col1, col2, = st.columns(2)

soma_por_classe = df_reg.groupby('Class')['Amount'].sum()
soma_por_classe.index = ['Não Fraude', 'Fraude']
percentis = px.pie(soma_por_classe, values='Amount', names=soma_por_classe.index, title='Análise de Perdas Financeiras por Fraude')
col2.plotly_chart(percentis)

teste = df_reg['Class'].value_counts()
ocorrencias_por_classe = df_reg['Class'].value_counts()
graf_fraudes = px.bar(x=ocorrencias_por_classe.index, y=ocorrencias_por_classe.values, title='Ocorrências de Fraudes')
graf_fraudes.update_layout(xaxis_title='Fraude', yaxis_title='Número de Ocorrências')
col1.plotly_chart(graf_fraudes)

chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["col1", "col2"])
st.scatter_chart(
    df_reg,
    x='Time',
    y='Amount'
)
