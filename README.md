# Classificação de Tumores 

Projeto académico de Machine Learning para comparar algoritmos de classificação no dataset **Breast Cancer Wisconsin (Diagnostic)**.

O objetivo é comparar os modelos K-Nearest Neighbors (KNN), Random Forest e Support Vector Machine (SVM), usando calibração de hiperparâmetros e validação cruzada estratificada de 10 folds.

## Dataset

- Fonte: [Kaggle — Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Arquivo local: `base/data.csv`
- Registros: 569
- Atributos preditivos: 30 variáveis numéricas contínuas
- Classes: benigno (`B`) e maligno (`M`)

Durante o carregamento, as colunas `id` e `Unnamed: 32`, quando presente, são descartadas. A variável-alvo é convertida para valores numéricos:

```text
B = 0  (Benigno)
M = 1  (Maligno)
```



## Instalação

No PowerShell, abra a pasta do projeto:

```powershell
cd ..\trabalho_IA_Classificacao
```

Crie o ambiente virtual:

```powershell
py -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```
Instale as dependências:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verificação do dataset

Para carregar e validar o CSV:

```powershell
python -m src.carregar_dados
```

O módulo `src/carregar_dados.py` também verifica se:

- o arquivo `base/data.csv` existe;
- a coluna `diagnosis` está presente;
- as classes são somente `B` e `M`;
- não existem valores ausentes;
- todos os atributos preditivos são numéricos.

## Análise exploratória

O notebook `notebooks/01_analise_exploratoria.ipynb` documenta a análise inicial da base, incluindo:

- dimensões, tipos e estatísticas descritivas;
- valores ausentes e linhas duplicadas;
- distribuição das classes;
- gráfico de distribuição dos diagnósticos;
- comparação das escalas dos atributos;
- correlação entre atributos e diagnóstico;
- mapa de calor dos 10 atributos mais correlacionados com a variável-alvo.

Para abri-lo, com o ambiente virtual ativo:

```powershell
jupyter lab --ServerApp.use_redirect_file=False
```

Copie para o navegador o endereço `http://localhost:...` exibido no terminal e abra o notebook em `notebooks/`.

## Resultados finais

Os modelos foram calibrados com `GridSearchCV` e avaliados com validação cruzada estratificada de 10 folds.

| Modelo | Acurácia média | Desvio padrão |
|---|---:|---:|
| SVM | **97,71%** | **1,58%** |
| KNN | 97,36% | 1,96% |
| Random Forest | 96,65% | 2,30% |

O SVM foi o modelo selecionado. Além de obter a maior acurácia média e o menor desvio padrão, classificou corretamente 203 dos 212 casos malignos e apresentou somente 9 falsos negativos.

As matrizes de confusão, curvas de aprendizagem e a tabela detalhada estão em `resultados/`.
