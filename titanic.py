import os
import argparse
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score

def create_pipeline(n_trees, numeric_features=["Age", "Fare"],\
    categorical_features=["Embarked", "Sex"], max_depth=None, max_features='sqrt'):

    ## Encoder les données imputées ou transformées.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=n_trees, max_features=max_features,
                                                max_depth=max_depth)),
        ]
    )
    return pipe

def evaluate(y_test, y_pred):
    rdmf_score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return rdmf_score, cm

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

TrainingData = pd.read_csv(os.path.join(BASE_PATH, "data.csv"))

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée
# une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train], axis = 1).to_csv("train.csv")
pd.concat([X_test, y_test], axis = 1).to_csv("test.csv")

# Random Forest
# Ici demandons d'avoir n_trees arbres
pipe = create_pipeline(n_trees)
pipe.fit(X_train, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset
# de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction

y_pred = pipe.predict(X_test)
rdmf_score, cm = evaluate(y_test, y_pred)

print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(cm)



