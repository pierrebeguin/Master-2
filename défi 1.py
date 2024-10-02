#!/usr/bin/env python
# coding: utf-8

# # importer data + séparation

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

display(data.head())

# exploration des données

data['BloodCulture'].value_counts()

# -> 10 fois plus de résultats négatif que de résultats positif


# In[2]:


# définir X et y (features et labels)

X = data.drop(columns=['ID','BloodCulture'])
y = data[['BloodCulture']]

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

import sklearn
import imblearn
! pip uninstall scikit-learn imbalanced-learn
! pip install scikit-learn imbalanced-learn

# Gestion du déséquilibre des classes
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
# In[3]:


# Gérer les valeurs manquantes
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# vérifier qu'il n'y a plus de valeur manquante

#X_train.isna().sum()
#X_test.isna().sum()

# Standardiser les données
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Modèle

# In[4]:


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Modèle régression logistique
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model.fit(X_train_scaled, y_train)


# Modèle K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # Choisir le nombre de voisins
knn_model.fit(X_train_scaled, y_train)


# Modèle Support Vector Machine
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', probability=True)  # Utilisation du noyau radial
svm_model.fit(X_train_scaled, y_train)


# Modèle forêts aléatoires
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)


# Modèle Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)


# Modèle Réseau de Neurones (MLP)
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(max_iter=500)  # Nombre d'itérations
nn_model.fit(X_train_scaled, y_train)


# # évaluation des modèles

# In[5]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Évaluation des modèles
models = {
    "Régression Logistique": lg_model,
    "K-Nearest Neighbors": knn_model,
    "Support Vector Machine": svm_model,
    "Forêts Aléatoires": rf_model,
    "Gradient Boosting": gb_model,
    "Réseau de Neurones": nn_model
}

# Boucle pour évaluer chaque modèle
for model_name, model in models.items():
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    
    # Évaluation
    print(f"\nÉvaluation du modèle : {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Matrice de Confusion:\n", confusion_matrix(y_test, y_pred))
    print("Rapport de Classification:\n", classification_report(y_test, y_pred))


# # courbe ROC

# In[6]:


# Importation des bibliothèques nécessaires
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialisation du graphique
plt.figure(figsize=(10, 8))

# Boucle pour évaluer chaque modèle
for model_name, model in models.items():
    # Prédictions de probabilité sur l'ensemble de test
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilité de la classe positive

    # Calcul de la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)  # FPR: Taux de faux positifs, TPR: Taux de vrais positifs
    roc_auc = auc(fpr, tpr)  # Calcul de l'AUC

    # Tracé de la courbe ROC
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Tracer la ligne de chance
plt.plot([0, 1], [0, 1], 'k--', label='AUC = 0.5 (Chance)')

# Étiquetage des axes
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbes ROC des Modèles')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




