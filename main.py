import os
import argparse
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.pipeline.build_pipeline import create_pipeline
from src.models.train_evaluate import train, evaluate

# Load environment variables
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--n_trees", type=int, default=20, help="number of trees")

args = parser.parse_args()
n_trees = args.n_trees
print(f"n_trees: {n_trees}")


jeton_api = os.environ.get("JETON_API", "")
if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")

BASE_PATH = os.environ.get("BASE_PATH", "")

TrainingData = pd.read_csv(os.path.join(BASE_PATH, "raw", "data.csv"))

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée
# une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(BASE_PATH, "derived", "train.csv"))
pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(BASE_PATH, "derived", "test.csv"))

pipe = create_pipeline(n_trees)
pipe = train(pipe, X_train, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset
# de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction

y_pred = pipe.predict(X_test)
rdmf_score, cm = evaluate(y_test, y_pred)

print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(cm)
