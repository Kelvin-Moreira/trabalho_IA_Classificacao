from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from src.carregar_dados import carregar_dados
from src.modelos import RANDOM_STATE, criar_modelos, obter_parametros

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_FIGURAS = RAIZ_PROJETO / "resultados" / "figuras"
PASTA_METRICAS = RAIZ_PROJETO / "resultados" / "metricas"
PROPORCAO_TESTE = 0.2
NUM_FOLDS = 10


def salvar_matriz_confusao(nome: str, y_reais, previsoes) -> None:
    """Gera e salva a matriz de confusão do conjunto de teste."""

    matriz = ConfusionMatrixDisplay.from_predictions(
        y_reais,
        previsoes,
        labels=[0, 1],
        display_labels=["Benigno", "Maligno"],
        cmap="Blues",
    )
    matriz.ax_.set_xlabel("Classe prevista")
    matriz.ax_.set_ylabel("Classe real")
    plt.title(f"Matriz de confusão — {nome} (teste)")
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / f"matriz_confusao_{nome.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def salvar_curva_aprendizagem(nome: str, modelo, X, y, validacao) -> None:
    """Gera e salva a curva de aprendizagem usando apenas os dados de treino."""

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
    """Calibra com 10 folds no treino e avalia uma vez no teste separado."""

    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=PROPORCAO_TESTE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    validacao = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)

    resultados = []
    melhores_modelos = {}

    for nome, modelo in criar_modelos().items():
        busca = GridSearchCV(
            estimator=modelo,
            param_grid=obter_parametros()[nome],
            scoring="accuracy",
            cv=validacao,
            n_jobs=-1,
        )
        busca.fit(X_treino, y_treino)

        melhor_modelo = busca.best_estimator_
        melhores_modelos[nome] = melhor_modelo

        salvar_curva_aprendizagem(
            nome, melhor_modelo, X_treino, y_treino, validacao
        )

        resultados.append({
            "modelo": nome,
            "melhores_parametros": str(busca.best_params_),
            "acuracia_media_cv": busca.best_score_,
            "desvio_padrao_cv": busca.cv_results_["std_test_score"][busca.best_index_],
        })

    # O teste só é consultado depois que todas as buscas foram concluídas.
    for resultado in resultados:
        nome = resultado["modelo"]
        previsoes = melhores_modelos[nome].predict(X_teste)
        resultado["acuracia_teste"] = accuracy_score(y_teste, previsoes)
        salvar_matriz_confusao(nome, y_teste, previsoes)

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
