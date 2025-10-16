#!/usr/bin/env python
# coding: utf-8

# Los conjuntos de datos contienen transacciones realizadas con tarjetas de crédito en septiembre de 2013 por titulares europeos.
# Este conjunto de datos presenta transacciones ocurridas en dos días, donde se registraron 492 fraudes de un total de 284.807 transacciones. El conjunto de datos presenta un alto desequilibrio: la clase positiva (fraudes) representa el 0,172 % del total de transacciones.
# 
# Contiene únicamente variables de entrada numéricas resultantes de una transformación PCA. Lamentablemente, debido a cuestiones de confidencialidad, no podemos proporcionar las características originales ni más información sobre los datos. Las características V1, V2, … V28 son los componentes principales obtenidos con PCA; las únicas características que no se han transformado con PCA son «Tiempo» e «Importe». La característica «Tiempo» contiene los segundos transcurridos entre cada transacción y la primera transacción del conjunto de datos. La característica «Importe» es el importe de la transacción; esta característica puede utilizarse para el aprendizaje sensible a los costes según el ejemplo. La característica «Clase» es la variable de respuesta y toma el valor 1 en caso de fraude y 0 en caso contrario.
# 
# Dada la tasa de desequilibrio de clases, recomendamos medir la precisión mediante el área bajo la curva de precisión-recuperación (AUPRC). La precisión de la matriz de confusión no es significativa para la clasificación desequilibrada.
# 
# El conjunto de datos se recopiló y analizó durante una colaboración de investigación entre Worldline y el Grupo de Aprendizaje Automático de la ULB (Universidad Libre de Bruselas) sobre minería de big data y detección de fraude.

# ## Objetivo:
# Predecir el fraude con tarjetas de crédito en los datos transaccionales. 
# 
#     Se Utilizara Logistic Regresion y TensorFlow para construir el modelo predictivo
#     
# Para esto primero haremos una exploración de los datos, despues una construcción de la red neuronal y por ultimo la visualización

# In[1]:


#importar librerias
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


# In[23]:


df = pd.read_csv("C:/Users/Jesus Eduardo/Documents/Bank marketing/Marketing bancario/creditcard.csv")


# In[24]:


#Echamos un promer vistazo a los datos 
df.head(10)


# In[25]:


print("El conjunto de datos de marketing bancario consta de {rows} filas.".format(rows = len(df)))


# In[26]:


#y el valor de datos faltantes sera:
missing_values = df.isnull().mean()*100

missing_values.sum()


# Como podemos ver es un dataset sin valores faltantes 
# 

# In[27]:


df.describe()


# In[28]:


number_of_fraud = len(df[df.Class == 1])
number_of_normal= len(df[df.Class == 0])

print ("Fraude:", number_of_fraud)
print ("Normal:",number_of_normal)

value_counts = df['Class'].value_counts()
value_counts.plot.bar(title = 'Fraude y normal')


# Fraude es una minima parte de los clientes

# Veamos cómo se compara el tiempo entre transacciones fraudulentas y normales.

# In[29]:


print ("Fraud")
print (df.Time[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Time[df.Class == 0].describe())


# In[30]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Tiempo (en segundos)')
plt.ylabel('Numero de transacciones')
plt.show()


# La función "Tiempo" es bastante similar en ambos tipos de transacciones. Se podría argumentar que las transacciones fraudulentas se distribuyen de forma más uniforme, mientras que las transacciones normales tienen una distribución cíclica. Esto nos ayudara a facilitar la detección de una transacción fraudulenta en horas de baja demanda.

# importe de la transacción entre fraude y normal client

# In[31]:


print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())


# In[32]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 30

ax1.hist(df.Amount[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Amount[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Importe ($)')
plt.ylabel('Numero de transacciones')
plt.yscale('log')
plt.show()


# In[33]:


df['Monto_máximo_de_fraude'] = 1
df.loc[df.Amount <= 2125.87, 'Monto_máximo_de_fraude'] = 0


# In[34]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')

plt.xlabel('Tiempo (en segundos)')
plt.ylabel('Importe')
plt.show()


# No se puede sacar nada de esto 
# 
# Entonces prosigamos con el analisis

# In[35]:


#Seleccionamos solo las columnas que contienen las caracteristicas v1, v1,...,v28
v_features = df.iloc[:,1:29].columns


# In[18]:


plt.figure(figsize=(12, 28 * 4))
gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    # Reemplaza sns.distplot con sns.histplot
    sns.histplot(df[cn][df.Class == 1], bins=50, kde=True, ax=ax, color='blue', label='Clase 1')
    sns.histplot(df[cn][df.Class == 0], bins=50, kde=True, ax=ax, color='red', label='Clase 0')
    ax.set_xlabel('')
    ax.set_title('Histograma de característica: ' + str(cn))
    ax.legend() # Agrega una leyenda para distinguir las clases
plt.tight_layout() # Ajusta automáticamente los subplots para que no se superpongan
plt.show()


# In[36]:


plt.figure(figsize=(12, len(v_features) * 4)) # Ajusta la altura de la figura dinámicamente
gs = gridspec.GridSpec(len(v_features), 1) # Ajusta el número de filas dinámicamente

for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])

    # Histograma para la Clase 0
    sns.histplot(
        df[cn][df.Class == 0],
        bins=50,
        kde=True,
        ax=ax,
        color='red',       # Color para la Clase 0
        label='Clase 0',
        alpha=0.6,         # Hace el histograma semi-transparente
        stat='density'     # Normaliza para comparar formas, no conteos
    )

    # Histograma para la Clase 1
    sns.histplot(
        df[cn][df.Class == 1],
        bins=50,
        kde=True,
        ax=ax,
        color='blue',      # Color para la Clase 1 (más visible)
        label='Clase 1',
        alpha=0.8,         # Un poco menos transparente para que resalte
        stat='density'     # Normaliza también para la Clase 1
    )

    ax.set_xlabel('')
    ax.set_title(f'Histograma de característica: {cn}') # Uso de f-string para un título más limpio
    ax.legend() # Muestra la leyenda para distinguir las clases

plt.tight_layout() # Ajusta automáticamente los subplots para que no se superpongan
plt.show()


# Podemos ver que nuestros datos estan desequilibrados, por lo que haremos algo al rescpeto, pero primero vamos a escalar los datos.

# ## standardscaler

# In[37]:


X=df.drop(columns=["Class"])
y=df["Class"]


# In[39]:


from sklearn import preprocessing


# In[41]:


names=X.columns
scaled_df = preprocessing.scale(X)
scaled_df = pd.DataFrame(scaled_df,columns=names)
scaled_df.head()


# Al observar las características de tiempo y cantidad, podemos afirmar que se han escalado.

# In[42]:


scaled_df[["Amount","Time"]].describe()


# Dado que aquí se trata del problema de un conjunto de datos desequilibrado, intentaremos equilibrarlo mediante la técnica "SMOTE". Este es otro método de sobremuestreo simple, pero en lugar de simplemente duplicar la clase minoritaria, se generan datos sintéticos.

# ## División de los datos entrenamiento y prueba

# Ahora dividiremos el conjunto de datos estandarizados en entrenamiento y prueba, luego haremos un sobremuestreo en el conjunto de datos de entrenamiento y luego haremos la clasificación en función del entrenamiento.

# In[43]:


X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size = 0.30, random_state = 0, shuffle = True, stratify = y)


# In[44]:


X_train.shape, X_test.shape


# In[45]:


y_train.value_counts()


# In[46]:


y_test.value_counts()


# ## Equilibrando datos con el metodo "SMOTE"

# In[3]:


from imblearn.over_sampling import SMOTE


# In[6]:


sm = SMOTE(random_state = 33)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())


# In[54]:


#Veamos si esta equilibrado
pd.Series(y_train_new).value_counts().plot(kind="bar")


# Como podemos ver ya se equilibraron los datos. Podemos empezar a crear modelos predictivos.

# ## Logistic Regression¶
# 

# In[56]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, solver = 'lbfgs')
clf.fit(X_train_new, y_train_new)
train_pred = clf.predict(X_train_new)
test_pred = clf.predict(X_test)


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print('Puntuación de precisión del conjunto de datos de entrenamiento = ', accuracy_score(train_pred, y_train_new))
print('Puntuación de precisión para el conjunto de datos de prueba = ', accuracy_score(test_pred, y_test))


# In[93]:


#Matriz de confusion
cm=confusion_matrix(y_test, test_pred)
cm


# In[62]:


plt.figure(figsize=(8,6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt = 'g', cmap="Reds", cbar = False)
plt.xlabel("etiqueta predictiva", size = 18)
plt.ylabel("Etiqueta verdadera", size = 18)
plt.title("Trazado de la matriz de confusión para el modelo de regresión logística", size = 20)


# In[ ]:





# In[65]:


print("El porcentaje de casos 'sin Fraude' clasificados erróneamente mediante regresión logística es:", (2018/85295)*100)
print("Porcentaje de casos de 'Fraude' con predicción errónea de regresión logística:", (13/148)*100)


# Para la primera clase hubo 2027 falsos positivos, para la segunda solo 13. Solo es muy poco, comparado al tamaño de las clases respectivamente como se púede ver en los porcentajes anteriores.

# In[94]:


# Recuperación (Recall / Sensibilidad):
recall = recall_score(y_test, test_pred)
print(f"Recuperación (Recall): {recall:.4f}")


# Exactitud (Accuracy):
accuracy = accuracy_score(y_test,test_pred)
print(f"Exactitud (Accuracy): {accuracy:.4f}")


# In[ ]:





# ## Uso de redes neuronales(Tensorflow)

# In[67]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# Arquitectura del modelo

# In[69]:


from tensorflow.keras.layers import Input 


# X_train_new.shape[1] te da el número de características en tu conjunto de datos.
input_shape = (X_train_new.shape[1],) 

model = Sequential()
# Añade la capa Input primero
model.add(Input(shape=input_shape))
# Ahora, la primera capa Dense ya no necesita input_dim
model.add(Dense(X_train_new.shape[1], activation = 'relu')) 
model.add(BatchNormalization())

model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

# Opcional: Imprime el resumen del modelo para verificar
model.summary()


# Los hiperparámetros que utilizamos son la Batch Normalization, y Dropout. La función de activación que utilizamos para las capas ocultas es "relu", y para la salida, la función "Sigmoide". También utilizamos dos capas ocultas con 64 unidades cada una.

# ## Optimizador

# In[71]:


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy')


# Para proteger nuestro modelo del sobreajuste, utilizaremos la función de detención temprana de Tensorflow. Esta función, una vez que la métrica de evaluación mencionada deja de mejorar, detendrá el número de épocas. También utilizamos una tasa de aprendizaje de 0,0001.

# In[73]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)


# In[74]:


history = model.fit(x=X_train_new, y=y_train_new, batch_size = 256, epochs=150,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])


# 57 bastaron para el modelo

# In[75]:


evaluation_metrics=pd.DataFrame(model.history.history)
evaluation_metrics.plot(figsize=(10,5))
plt.title("Pérdida tanto para el entrenamiento como para la validación", size = 20)


# In[81]:


y_pred = model.predict(X_test)
# Para una clasificación binaria con sigmoide, el umbral común es 0.5
# Esto devolverá 1 si la probabilidad es >= 0.5, y 0 en caso contrario.
y_pred_classes = (y_pred >= 0.5).astype(int)


# Ahora veamos la matriz de confusion resultante

# In[82]:


cm_nn=confusion_matrix(y_test, y_pred_classes)
cm_nn


# In[83]:


plt.figure(figsize=(8,6))
sns.set(font_scale=1.2)
sns.heatmap(cm_nn, annot=True, fmt = 'g', cmap="winter", cbar = False)
plt.xlabel("Etiqueta Predictiva", size = 18)
plt.ylabel("Etiqueta verdadera", size = 18)
plt.title("Trazado de la matriz de confusión para el modelo de red neuronal", size = 20)


# In[95]:


# Encontremos las puntuaciones de precisión y recuperación.
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred_classes)
print(f"Precisión (Precision): {precision:.4f}")

# Recuperación (Recall / Sensibilidad):
recall = recall_score(y_test, y_pred_classes)
print(f"Recuperación (Recall): {recall:.4f}")


# Exactitud (Accuracy):
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Exactitud (Accuracy): {accuracy:.4f}")


# Comparando ambos modelos: el problema esta en que ofrece una predicción muy buena para la clase mayoritaria (0= casos "sin fraude"), pero para la clase minoritaria (1= casos de "fraude"), su rendimiento es ligeramente inferior al de la regresión logística. Sin embargo, ajustando un poco más los hiperparámetros, el modelo podrá tener un mejor rendimiento que la regresión logística incluso para la clase minoritaria.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




