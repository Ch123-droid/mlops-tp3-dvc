import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Chargement des paramètres d'entraînement
p = yaml.safe_load(open('params.yaml', encoding='utf-8'))['entrainement']

# Lecture des données d'entraînement (entrée : sortie de l'étape 1)
train = pd.read_csv('data/prepare/train.csv')
X, y = train.drop(columns='fraude'), train['fraude']

# Entraînement du modèle Random Forest
modele = RandomForestClassifier(
    n_estimators=p['n_estimators'],
    max_depth=p['max_depth'],
    random_state=p['random_state']
).fit(X, y)

# Création du dossier pour le modèle si nécessaire
os.makedirs('models', exist_ok=True)

# Exportation de l'artefact modèle
joblib.dump(modele, 'models/model.pkl')

print('entrainement : modele ecrit dans models/model.pkl')