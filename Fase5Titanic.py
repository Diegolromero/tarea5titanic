#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[264]:


pd.read_csv('tested.csv')
df = pd.read_csv('tested.csv')


# In[265]:


x_encoded = pd.get_dummies(x, drop_first=True)


# In[266]:


df.head()


# In[267]:


df = df.drop('PassengerId', axis=1)


# In[268]:


df.drop(['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket' ], axis=1, inplace=True)


# In[269]:


df = df.drop('Fare', axis=1)
df = df.drop('Parch', axis=1)
df = df.drop('SibSp', axis=1)


# In[270]:


df.head()


# In[271]:


data_types = df.dtypes
categorical_columns = data_types[data_types == 'object'].index
print(categorical_columns)


# In[272]:


df = df.dropna()


# In[273]:


import matplotlib.pyplot as plt


# Contar la cantidad de sobrevivientes y fallecidos
count_survived = df['Survived'].value_counts()[1]
count_died = df['Survived'].value_counts()[0]

# Crear el gráfico de histograma
plt.bar(['Sobrevivientes', 'Fallecidos'], [count_survived, count_died])
plt.xlabel('Estado')
plt.ylabel('Cantidad')
plt.title('Cantidad de sobrevivientes y fallecidos')
plt.show()


# In[253]:


#separar caracteristicas de la variable objetivo
x = df.drop('Survived', axis= 1)
y = df['Survived']

#dividir entrenamiento de prueba

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('precision del modelo: ',accuracy)

print("el porcentaje de precision del modelo es de: ",(accuracy*100),"%")


# In[254]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

features = ['Survived', 'Pclass', 'Age' ]
target = 'Survived'

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

con_mat = confusion_matrix(y_test, y_pred)

print(con_mat)


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(con_mat, annot=True, cmap="Blues", fmt="d", xticklabels=["negativo", "positivo"], yticklabels=["negativo", "positivo"])

ax.set_xlabel('predicción')
ax.set_ylabel('valor real')
ax.set_title('Matriz de Confusión')

plt.show()


# In[255]:


pip install graphviz


# In[256]:


import graphviz


# In[257]:


#separar caracteristicas de la variable objetivo
x = df.drop('Survived', axis= 1)
y = df['Survived']


#dividir entrenamiento de prueba

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


#crear instancia del modelo

arbol_decision = DecisionTreeClassifier(random_state=1)


#entrner modelo con el conjunto


arbol_decision.fit(x_train, y_train)


#UTILIZAR EL MODELO PARA HACER CONJUNTO DE PRUEBAS

y_pred = arbol_decision.predict(x_test)


# In[47]:


#EVALUAR DESEMPEÑO DEL MODELO 
accuracy = accuracy_score(y_test, y_pred)
print("la precision del modelo es: ", accuracy)


entre=accuracy*100
print("porcentaje",entre)


from sklearn.tree import DecisionTreeClassifier

arbol_decision = DecisionTreeClassifier(max_depth=3, min_samples_split=10)

arbol_decision.fit(x_train, y_train)



from sklearn.tree import export_graphviz
import graphviz

# Exportar el árbol de decisiones
export_graphviz(arbol_decision, out_file='arbol_decision.dot', feature_names=x.columns.values, filled=True, rounded=True, special_characters=True)


# Convertir el archivo .dot a un objeto gráfico
with open('arbol_decision.dot') as f:
    dot_graph = f.read()

graph = graphviz.Source(dot_graph)

# Mostrar y guardar el gráfico
graph


from IPython.display import Image
from graphviz import render

#render
render('dot', 'png', 'arbol_decision.dot')


# In[258]:


graph


# In[259]:


import matplotlib.pyplot as plt


survival_rate = df.groupby('Pclass')['Survived'].mean()

# Crear el gráfico de barras
plt.bar(survival_rate.index, survival_rate.values)
plt.xlabel('Clase de pasajero')
plt.ylabel('Proporción de supervivencia')
plt.title('Proporción de supervivencia por clase de pasajero')
plt.show()


# In[260]:


plt.scatter(df['Age'], df['Survived'])
plt.xlabel('Edad')
plt.ylabel('Supervivencia')
plt.title('Relación entre edad y supervivencia')
plt.show()


# In[261]:


plt.hist(df['Age'], bins=10)
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de la edad')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




