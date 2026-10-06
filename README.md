# Classificação de Tumores 

Projeto académico de Machine Learning para comparar algoritmos de classificação no dataset **Breast Cancer Wisconsin (Diagnostic)**.

O objetivo é comparar os modelos K-Nearest Neighbors (KNN), Random Forest e Support Vector Machine (SVM), usando calibração de hiperparâmetros, validação cruzada estratificada de 10 folds e um conjunto de teste independente.

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

## Avaliação dos modelos

Antes da calibração, a base é dividida de forma estratificada em 455 registros para treino (80%) e 114 para teste (20%). No treino, o `GridSearchCV` calibra cada modelo com validação cruzada estratificada de 10 folds. As grades de hiperparâmetros estão em `src/modelos.py`. O modelo é selecionado pela acurácia média da validação cruzada; o teste não participa dessa escolha.

As curvas de aprendizagem usam apenas os dados de treino. Depois da calibração, cada modelo é avaliado uma vez no conjunto de teste; as matrizes de confusão mostram somente essas previsões. Para executar:

```powershell
python -m src.avaliacao
```

O comando atualiza `resultados/metricas/comparacao_modelos.csv` com a acurácia média e o desvio padrão dos 10 folds, os melhores hiperparâmetros e a acurácia no teste. Também atualiza as matrizes de confusão e as curvas de aprendizagem em `resultados/figuras/`.

## Resultados

| Modelo | Acurácia média nos 10 folds | Desvio padrão | Acurácia no teste |
|---|---:|---:|---:|
| SVM | **97,36%** | 1,91 p.p. | **100,00%** |
| KNN | 96,48% | 1,77 p.p. | 98,25% |
| Random Forest | 96,48% | 2,46 p.p. | 97,37% |

Os melhores hiperparâmetros encontrados foram:

- **SVM:** núcleo RBF, `C=10` e `gamma=0,01`;
- **KNN:** 3 vizinhos e pesos uniformes;
- **Random Forest:** 100 árvores, profundidade sem limite e mínimo de 2 amostras para dividir um nó.

No conjunto de teste havia 72 registros benignos e 42 malignos. A SVM classificou corretamente os 114 registros, sem falsos positivos ou falsos negativos. O KNN acertou 112 registros, com um falso positivo e um falso negativo. A Random Forest acertou 111, com dois falsos positivos e um falso negativo.

As curvas de aprendizagem mostram acurácias de treino e validação próximas para SVM e KNN. A Random Forest atingiu 100% de acurácia no treino em todos os tamanhos avaliados, mas ficou abaixo disso na validação, indicando maior ajuste aos dados de treino. Pela maior acurácia média nos 10 folds, a SVM foi o modelo selecionado. Seu resultado de 100% refere-se apenas aos 114 registros deste teste e não garante o mesmo desempenho em novos dados.
