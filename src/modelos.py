from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 69


def criar_modelos() -> dict[str, Pipeline]:
    """Cria os modelos que serão comparados."""

    return {
        "KNN": Pipeline([
            ("normalizador", StandardScaler()), # StandardScaler é usado para normalizar os dados rapaziada
            ("modelo", KNeighborsClassifier()),
        ]),
        "Random Forest": Pipeline([
            (
                "modelo",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "SVM": Pipeline([
            ("normalizador", StandardScaler()),
            ("modelo", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def obter_parametros() -> dict[str, dict]:
    """Retorna os parâmetros que serão calibrados."""

    return {
        "KNN": {
            "modelo__n_neighbors": [3, 5, 7, 9],
            "modelo__weights": ["uniform", "distance"],
        },
        "Random Forest": {
            "modelo__n_estimators": [100, 200],
            "modelo__max_depth": [None, 10, 20],
            "modelo__min_samples_split": [2, 5],
        },
        "SVM": {
            "modelo__C": [0.1, 1, 10],
            "modelo__kernel": ["linear", "rbf"],
            "modelo__gamma": ["scale", 0.001, 0.01, 0.1],
        },
    }
