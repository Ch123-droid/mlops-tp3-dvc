import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Mode sans interface graphique pour les serveurs/scripts CI/CD
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             ConfusionMatrixDisplay)

# Chargement du jeu de test et du modèle entraîné
test = pd.read_csv('data/prepare/test.csv')
X, y = test.drop(columns='fraude'), test['fraude']
modele = joblib.load('models/model.pkl')

# Calcul des prédictions
pred = modele.predict(X)

# Calcul des métriques de performance
mets = {
    'f1': round(f1_score(y, pred), 4),
    'precision': round(precision_score(y, pred), 4),
    'recall': round(recall_score(y, pred), 4)
}

# Création du dossier de rapports si nécessaire
os.makedirs('reports', exist_ok=True)

# Sauvegarde des métriques au format JSON (sortie 1)
with open('reports/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(mets, f, indent=2)

# Génération et sauvegarde de la matrice de confusion (sortie 2)
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.savefig('reports/confusion.png', dpi=110, bbox_inches='tight')
plt.close()

print('evaluation :', mets)