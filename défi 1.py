#!/usr/bin/env python
# coding: utf-8

# In[1]:


# le fichier est mis sur github pour y avoir accès depuis n'importe quelle pc

import pandas as pd
import requests
from io import StringIO

lien_csv = "https://raw.githubusercontent.com/pierrebeguin/Master-2/main/bacteriemie.csv"
response = requests.get(lien_csv)
data = pd.read_csv(StringIO(response.text), sep=",")

# Changer les 'no' par '0' et les 'yes' par '1'
data['BloodCulture'] = data['BloodCulture'].replace({'no': 0, 'yes': 1})

data.head()


# In[3]:


# définir X et y

X = data.drop(columns=['ID','BloodCulture'])
y = data[['BloodCulture']]

# vérifier que x et y fonctionne

display(X.head())
display(y.head())


# In[4]:


# Séparation des données en 2

from sklearn.model_selection import train_test_split

# Divisez les données en ensembles d'entraînement et de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# vérifier la taille
print('taille du train :')
print('taille de X_train = ',X_train.shape)
print('taille de y_train = ',y_train.shape)
print(' ')
print('taille du test :')
print('taille de X_test = ',X_test.shape)
print('taille de y_test = ',y_test.shape)


# In[7]:


# Gérer les valeurs manquantes
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# vérifier qu'il n'y a plus de valeur manquante

X_train.isna().sum()
X_test.isna().sum()


# In[11]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_scaled, y_train)

# Évaluer le modèle
print('Train score:', model.score(X_train_scaled, y_train))
print('Test score:', model.score(X_test_scaled, y_test))


# In[11]:


# Convertir les tableaux NumPy en DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns) 

# Afficher X_train après standardisation
print('X_train après standardisation :')
display(X_train_scaled_df)

# Afficher X_test après standardisation
print('X_test après standardisation :')
display(X_test_scaled_df)


# In[ ]:





# In[9]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X = X_train_scaled_df.values
y = y_train

# Créer le modèle de régression Logistic
model = LogisticRegression()

# Ajuster le modèle aux données
model.fit(X, y)

# Prédictions sur les données d'entraînement
y_pred = model.predict(X)

# Tracer le nuage de points avec la régression linéaire
plt.figure(figsize=(10, 6))  # Ajuster la taille de la figure
plt.scatter(y, y_pred, color='blue', alpha=0.5)  # Nuage de points
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('y_train (réel)')
plt.ylabel('y_pred (prédiction)')
plt.title('Régression Logistic entre y_train et les variables indépendantes')
plt.grid(True)
plt.axis('equal')
plt.show()


# In[ ]:




