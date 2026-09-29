from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
)
from src.carregar_dados import carregar_dados
from src.modelos import RANDOM_STATE, criar_modelos, obter_parametros

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_FIGURAS = RAIZ_PROJETO / "resultados" / "figuras"
PASTA_METRICAS = RAIZ_PROJETO / "resultados" / "metricas"


def salvar_matriz_confusao(nome: str, modelo, X, y, validacao) -> float:
    """Gera e salva a matriz de confusão com previsões dos 10 folds."""

    previsoes = cross_val_predict(modelo, X, y, cv=validacao, n_jobs=-1)

    matriz = ConfusionMatrixDisplay.from_predictions(
        y,
        previsoes,
        display_labels=["Benigno", "Maligno"],
        cmap="Blues",
    )
    matriz.ax_.set_xlabel("Classe prevista")
    matriz.ax_.set_ylabel("Classe real")
    plt.title(f"Matriz de confusão — {nome}")
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / f"matriz_confusao_{nome.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

    return accuracy_score(y, previsoes)


def salvar_curva_aprendizagem(nome: str, modelo, X, y, validacao) -> None:
    """Gera e salva a curva de aprendizagem do modelo."""

    tamanhos, scores_treino, scores_validacao = learning_curve(
        modelo,
        X,
        y,
        cv=validacao,
        scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(tamanhos, scores_treino.mean(axis=1), marker="o", label="Treino")
    plt.plot(tamanhos, scores_validacao.mean(axis=1), marker="o", label="Validação")
    plt.title(f"Curva de aprendizagem — {nome}")
    plt.xlabel("Quantidade de registros de treino")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / f"curva_aprendizagem_{nome.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def avaliar_modelos() -> pd.DataFrame:
    """Calibra e avalia KNN, Random Forest e SVM com 10 folds."""

    X, y = carregar_dados()
    validacao = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)

    resultados = []

    for nome, modelo in criar_modelos().items():
        busca = GridSearchCV(
            estimator=modelo,
            param_grid=obter_parametros()[nome],
            scoring="accuracy",
            cv=validacao,
            n_jobs=-1,
        )
        busca.fit(X, y)

        indice = busca.best_index_
        melhor_modelo = busca.best_estimator_

        acuracia_previsoes = salvar_matriz_confusao(
            nome,
            melhor_modelo,
            X,
            y,
            validacao,
        )
        salvar_curva_aprendizagem(nome, melhor_modelo, X, y, validacao)

        resultados.append({
            "modelo": nome,
            "melhores_parametros": str(busca.best_params_),
            "acuracia_media_cv": busca.cv_results_["mean_test_score"][indice],
            "desvio_padrao_cv": busca.cv_results_["std_test_score"][indice],
            "acuracia_previsoes_cv": acuracia_previsoes,
        })

    tabela_resultados = pd.DataFrame(resultados).sort_values(
        "acuracia_media_cv",
        ascending=False,
    )
    tabela_resultados.to_csv(
        PASTA_METRICAS / "comparacao_modelos.csv",
        index=False,
    )

    return tabela_resultados


if __name__ == "__main__":
    print(avaliar_modelos().to_string(index=False))
