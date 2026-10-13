import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des paramètres de préparation
p = yaml.safe_load(open('params.yaml', encoding='utf-8'))['preparation']

# Lecture du jeu de données
df = pd.read_csv('data/transactions.csv')

# Découpage train / test
train, test = train_test_split(
    df, 
    test_size=p['test_size'], 
    random_state=p['random_state']
)

# Création du dossier d'accueil si nécessaire
os.makedirs('data/prepare', exist_ok=True)

# Exporter les fichiers préparés
train.to_csv('data/prepare/train.csv', index=False)  # sortie 1
test.to_csv('data/prepare/test.csv', index=False)   # sortie 2

print('preparation :', len(train), 'lignes train /', len(test), 'lignes test')