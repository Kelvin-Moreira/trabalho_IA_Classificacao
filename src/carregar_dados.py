from pathlib import Path

import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_DATASET = RAIZ_PROJETO / "base" / "data.csv"

COLUNAS_DESCARTADAS = ["id", "Unnamed: 32"]
MAPEAMENTO_DIAGNOSTICO = {
    "B": 0,
    "M": 1,
}


def carregar_dados() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega e valida o dataset de câncer de mama."""

    if not CAMINHO_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {CAMINHO_DATASET}"
        )

    df = pd.read_csv(CAMINHO_DATASET)

    if "diagnosis" not in df.columns:
        raise ValueError("A coluna-alvo 'diagnosis' não foi encontrada.")

    df = df.drop(columns=COLUNAS_DESCARTADAS, errors="ignore")

    classes_encontradas = set(df["diagnosis"].dropna().unique())
    classes_esperadas = set(MAPEAMENTO_DIAGNOSTICO)

    if not classes_encontradas.issubset(classes_esperadas):
        raise ValueError(
            f"Classes inesperadas: {sorted(classes_encontradas)}"
        )

    y = df["diagnosis"].map(MAPEAMENTO_DIAGNOSTICO)
    X = df.drop(columns="diagnosis")

    if y.isna().any():
        raise ValueError("Existem valores ausentes ou inválidos no diagnóstico.")

    if X.isna().any().any():
        colunas_com_ausentes = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Existem valores ausentes em: {colunas_com_ausentes}"
        )

    if not all(pd.api.types.is_numeric_dtype(X[coluna]) for coluna in X):
        raise TypeError("Existem atributos preditivos que não são numéricos.")

    return X, y


if __name__ == "__main__":
    atributos, alvo = carregar_dados()

    print(f"Quantidade de registros: {len(atributos)}")
    print(f"Quantidade de atributos: {atributos.shape[1]}")
    print("\nDistribuição das classes:")
    print(alvo.value_counts().sort_index())