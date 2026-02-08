import os
import duckdb
import logging
import argparse

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from titanicml.pipeline.build_pipeline import create_pipeline
from titanicml.models.train_evaluate import train, evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--n_trees", type=int, default=20, help="number of trees")

args = parser.parse_args()
n_trees = args.n_trees
logging.info(f"n_trees: {n_trees}")


jeton_api = os.environ.get("JETON_API", "")

if jeton_api.startswith("$"):
    logging.info("API token has been configured properly")
else:
    logging.info("API token has not been configured")

# chemins
URL_RAW = os.getenv("URL_RAW")

data_path = os.environ.get("data_path", URL_RAW)

query =f"SELECT * FROM read_parquet('{data_path}');"
TrainingData = duckdb.sql(query).to_df()

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée
# une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(BASE_PATH, "derived", "train.csv"))
# pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(BASE_PATH, "derived", "test.csv"))

pipe = create_pipeline(n_trees)
pipe = train(pipe, X_train, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset
# de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction

y_pred = pipe.predict(X_test)
rdmf_score, cm = evaluate(y_test, y_pred)

logging.info(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
logging.info(20 * "-")
logging.info("matrice de confusion")
logging.info(cm)
