# Projeto de Machine Learning: Clusterização + Regressão

Este projeto usa o **Insurance Cost Dataset** para comparar duas estratégias de previsão de custos de seguro médico:

1. Regressão global, com um único modelo para todo o dataset.
2. Regressão por cluster, onde os dados são agrupados antes e depois cada grupo recebe um modelo próprio.

## Objetivo

Verificar se separar os dados em grupos antes da regressão melhora a previsão de `charges`.

## Estrutura

- `dataset.csv`: base bruta fornecida no enunciado.
- `main.py`: pipeline completo de limpeza, clusterização, regressão, avaliação e predição final.
- `artifacts/`: arquivos gerados automaticamente após a execução.

## Bibliotecas usadas

- `pandas` e `numpy` para manipulação de dados.
- `scikit-learn` para encoding, normalização, clusterização e regressão.
- `matplotlib` para visualizações.
- `joblib` para salvar o conjunto treinado.

## Como os dados foram preparados

O script faz a limpeza do CSV, remove duplicatas, padroniza textos para minúsculas e converte os tipos numéricos.

As transformações seguem a regra do enunciado:

- `sex` e `smoker` usam **Label Encoding**.
- `region` usa **One-Hot Encoding**.
- Os dados são normalizados com **StandardScaler** antes da clusterização.

O dataset tratado é salvo em `artifacts/dataset_tratado.csv`.

## Método de clusterização

O método escolhido foi **K-Means**, porque é simples, interpretável e funciona bem para separar perfis como:

- fumantes com custo alto;
- não fumantes com custo mais baixo;
- grupos intermediários por idade e IMC.

O número de clusters é escolhido automaticamente com base em:

- **Elbow Method** para observar a inércia.
- **Silhouette Score** para medir a qualidade da separação.

Nesta versão, o K-Means é ajustado na base completa antes da divisao treino/teste, e os rótulos de cluster são usados depois na regressão com split estratificado.

As figuras são salvas em:

- `artifacts/elbow.png`
- `artifacts/silhouette.png`
- `artifacts/clusters_pca.png`
- `artifacts/cluster_age_charges.png`
- `artifacts/cluster_bmi_charges.png`
- `artifacts/cluster_charges_boxplot.png`

## Modelo de regressão usado

O projeto compara três modelos permitidos na atividade:

- `LinearRegression`
- `KNeighborsRegressor`
- `DecisionTreeRegressor`

Cada modelo é avaliado em dois cenários:

- regressão global;
- regressão por cluster.

## Métricas

As métricas calculadas são:

- `MAE`
- `MSE`
- `RMSE`
- `R²`

Os resultados ficam salvos em `artifacts/metrics.json`.

## Comparação dos resultados

O script compara:

- regressão global vs regressão por cluster;
- menor erro em `RMSE`;
- diferença entre os dois cenários.
- resultado entre Linear, KNN e Tree.

### Resultado obtido nesta base

Na execução realizada neste dataset, o melhor número de clusters foi `k = 4`.

- Melhor modelo global: `LinearRegression`, `RMSE = 3846.20`
- Melhor modelo por cluster: `LinearRegression`, `RMSE = 3855.50`

Conclusão prática desta base: a regressão global ficou levemente melhor, então a clusterização não trouxe ganho de erro neste recorte específico. Entre os modelos testados, o Linear foi o melhor para esta base.

## Interpretação dos clusters

O projeto gera um resumo automático dos grupos em:

- `artifacts/cluster_profiles.csv`
- `artifacts/cluster_interpretation.txt`

Esse resumo ajuda a identificar padrões como:

- clusters com mais fumantes;
- clusters com custo médio maior;
- clusters com perfil mais jovem;
- clusters com IMC acima da média.

## Predição final

O script também aceita novos dados do usuário e retorna:

- o cluster identificado;
- o custo previsto do seguro.

Exemplo de entrada:

```text
age = 40
sex = male
bmi = 30
children = 2
smoker = yes
region = northwest
```

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

Ao final da execução o projeto salva o bundle treinado em:

- `artifacts/insurance_cluster_bundle.joblib`

## Integrantes

Henrique Estrela Santos  

## Conclusão final

Nesta base, a melhor abordagem foi a regressão global com `LinearRegression`, porque apresentou RMSE menor do que a regressão por cluster. A clusterização ainda é útil para interpretar perfis e segmentar grupos, mas, neste dataset e com esta configuração, não melhorou a precisão da previsão.
