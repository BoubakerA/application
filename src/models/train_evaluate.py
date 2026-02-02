from sklearn.metrics import confusion_matrix, accuracy_score


def train(pipe, X_train, y_train):
    # Random Forest
    # Ici demandons d'avoir n_trees arbres
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(y_test, y_pred):
    rdmf_score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return rdmf_score, cm
