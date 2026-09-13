# Análise de Clusterização — Segmentação de Clientes que Cancelaram (Churn)

**Objetivo:** testar diferentes algoritmos de clusterização (k de 2 a 7) sobre os clientes que
cancelaram (`Saiu == 1`), usando as variáveis `PontuacaoCredito`, `Idade`, `TempoRelacionamento`,
`Saldo`, `NumeroProdutos` e `SalarioEstimado` (padronizadas), para identificar perfis distintos
de churn e validar qual número de clusters melhor representa a estrutura dos dados.

---

## 1. Glossário das métricas

| Métrica | O que mede | Direção ideal | Observações |
|---|---|---|---|
| **Inertia** | Soma das distâncias quadráticas de cada ponto ao centróide do seu cluster | Quanto menor, melhor (mas sempre cai com mais clusters) | Usada só para o "método do cotovelo" — não comparar valor absoluto entre algoritmos diferentes |
| **Silhouette Score** | Quão bem separado e coeso cada ponto está em relação ao seu cluster vs. outros clusters | Quanto maior, melhor (varia de -1 a 1) | > 0.5 = estrutura forte; 0.25–0.5 = estrutura fraca/moderada; < 0.25 = pouca separação real |
| **Davies-Bouldin Index** | Razão entre dispersão interna dos clusters e distância entre eles | Quanto menor, melhor | Sensível a clusters de tamanhos/densidades muito diferentes |
| **Calinski-Harabasz Index** | Razão entre dispersão entre clusters e dispersão dentro dos clusters | Quanto maior, melhor | Tende a favorecer clusters convexos e de tamanho parecido |
| **Estabilidade (ARI médio via bootstrap)** | Quão consistente é o agrupamento ao re-rodar o algoritmo em subamostras dos dados | Quanto mais perto de 1, melhor | Desvio-padrão alto = solução instável, sensível a qual subconjunto de dados é usado |

---

## 2. K-Means

### 2.1 Setup

- Pré-processamento: `StandardScaler` nas 6 variáveis numéricas, ajustado em `df_treino_cancelamento`
- `KMeans(n_init=10, random_state=42)`, testado para k = 2 a 8
- Estabilidade: bootstrap com 30 reamostragens, 80% dos dados por rodada, comparação via Adjusted Rand Index (ARI) contra o clustering na base completa

### 2.2 Resultados

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|---|
| 2 | 7368.29 | 0.1723 | 2.1604 | 229.54 | 0.6414 | 0.4466 |
| 3 | 6483.43 | 0.1781 | 1.8807 | 227.45 | 0.5361 | 0.2889 |
| 4 | 5752.68 | 0.1620 | 1.8257 | 230.98 | **0.8696** | **0.1211** |
| 5 | 5343.00 | 0.1430 | 1.6733 | 213.64 | 0.5981 | 0.0818 |
| 6 | 5024.91 | 0.1419 | 1.5865 | 199.57 | 0.5689 | 0.1151 |
| 7 | 4768.35 | 0.1439 | 1.5780 | 187.86 | 0.5082 | 0.0780 |
| 8 | 4526.41 | 0.1453 | 1.5987 | 180.34 | 0.5151 | 0.0868 |

### 2.3 Interpretação

- **Silhouette** fica baixo em todos os k's testados (máximo 0.178, em k=3), o que indica que,
  com essas 6 variáveis, os clientes que cancelaram **não formam grupos claramente separados**
  — há sobreposição relevante entre os clusters independente do k escolhido.
- **Davies-Bouldin** melhora (cai) progressivamente até k=6-7, sugerindo que aumentar o número
  de clusters reduz a sobreposição relativa entre eles, mas o ganho marginal diminui depois de k=4.
- **Calinski-Harabasz** atinge o pico em k=4 (230.98), muito próximo de k=2, e cai de forma
  consistente a partir daí — não aponta para valores altos de k.
- **Estabilidade** é o dado mais decisivo: k=4 se destaca isoladamente (ARI médio de 0.87,
  desvio de apenas 0.12), enquanto os demais valores de k (incluindo k=3, com ARI de 0.54 e
  desvio de 0.29) mostram soluções bem menos consistentes entre reamostragens.
- **Conclusão parcial (k-means):** k=3 não é claramente ruim, mas também não se destaca — vence
  o silhouette por margem pequena e perde nas demais métricas. **k=4** é o candidato mais forte,
  por reunir estabilidade muito superior e métricas de separação comparáveis ou melhores.
- **Candidatos selecionados para classificação: k=3 e k=4.**
- **Ressalva geral:** como o silhouette é baixo mesmo no melhor k, vale considerar, em versões
  futuras, incluir variáveis adicionais (categóricas/comportamentais) ou testar algoritmos que
  lidam melhor com fronteiras pouco nítidas entre grupos.

---

## 3. K-Medoids

### 3.1 Setup

- Mesma base (`df_treino_cancelamento`) e mesmo `StandardScaler` (ajustado separadamente para
  essa base, não reaproveitado do `df_treino` completo).
- `KMedoids(method='alternate', init='k-medoids++', random_state=42)`, testado para k = 2 a 7.
- Estabilidade calculada com a mesma lógica de bootstrap + ARI usada no k-means.

### 3.2 Resultados

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|
| 2 | 0.1174 | 2.6777 | 184.02 | 0.0655 | 0.0806 |
| 3 | 0.0971 | 2.3613 | 159.32 | 0.1235 | 0.0668 |
| 4 | 0.1029 | 2.0737 | 160.24 | 0.1622 | 0.0501 |
| 5 | 0.1159 | 1.8890 | 164.82 | 0.2337 | 0.0650 |
| 6 | 0.1091 | 1.7759 | 158.50 | **0.2615** | 0.0519 |
| 7 | **0.1200** | **1.7710** | 157.93 | 0.2603 | 0.0554 |

### 3.3 Interpretação

- **Qualidade geral bem mais fraca que o k-means em todas as métricas.** Silhouette (0.097–0.120)
  fica abaixo da faixa do k-means (0.14–0.18) — usar um cliente real como representante do cluster,
  em vez do centro médio calculado, não ajudou a separar melhor os grupos.
- **Davies-Bouldin** melhora de forma consistente com k maior, de 2.68 (k=2) até 1.77 (k=6/k=7).
- **Calinski-Harabasz** tem pico isolado em k=2 (184.02), caindo e estabilizando entre 157–165
  do k=3 em diante, sem um segundo pico relevante.
- **Estabilidade é a diferença mais marcante frente ao k-means**: o teto aqui é 0.26 (k=6/k=7),
  muito abaixo do 0.87 do k-means em k=4. k=2, apesar do melhor Calinski-Harabasz, tem estabilidade
  quase nula (0.065 — praticamente aleatória), o que descarta esse k apesar do CH isolado.
- **Candidatos selecionados para classificação: k=6 e k=7** — reúnem o melhor equilíbrio entre
  Davies-Bouldin (empatados como melhores do grupo), estabilidade (as duas mais altas) e
  silhouette (k=7 é o melhor do grupo; k=6 logo atrás).
- **Comparação preliminar com k-means:** o K-Medoids parece se ajustar pior à estrutura desses
  dados — todas as métricas são mais fracas, principalmente a estabilidade. O k-means segue como
  referência mais forte entre os algoritmos testados até aqui, mas vale confirmar isso quando
  k=6/k=7 do K-Medoids forem testados nos classificadores (lembrando que k=4 do k-means, apesar
  de "vencedor" na clusterização, só se mostrou útil como feature preditiva no XGBoost).

---

## 4. Agglomerative Clustering (Clustering Hierárquico)

### 4.1 Setup

- Mesma base (`df_treino_cancelamento`) e mesmo `StandardScaler`.
- `AgglomerativeClustering(linkage='ward')`, testado para k = 2 a 7.
- Estabilidade calculada com a mesma lógica de bootstrap + ARI (sem `random_state`, já que o
  algoritmo é determinístico dado o dataset).

### 4.2 Resultados

| k | Linkage | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|---|
| 2 | ward | 0.1638 | 2.1841 | 191.81 | 0.1598 | 0.2372 |
| 3 | ward | 0.1616 | 1.9988 | **212.37** | **0.6574** | 0.1302 |
| 4 | ward | 0.1189 | 2.0308 | 192.90 | 0.4227 | 0.0797 |
| 5 | ward | 0.1018 | 1.8302 | 177.07 | 0.4385 | 0.0687 |
| 6 | ward | 0.0909 | **1.7268** | 160.20 | 0.4431 | 0.0648 |
| 7 | ward | 0.0967 | 1.8473 | 149.76 | 0.3797 | 0.0576 |

### 4.3 Interpretação

- **k=3 se destaca isoladamente como o melhor k do algoritmo**: maior estabilidade da tabela
  (ARI de 0.657, bem à frente de qualquer outro k), maior Calinski-Harabasz (212.37) e segundo
  melhor silhouette (0.162, praticamente empatado com k=2).
- **k=2, apesar do melhor silhouette (0.164) e CH razoável, é descartado**: sua estabilidade
  (0.160) é baixa e o desvio-padrão (0.237) é maior que a própria média — sinal de resultado
  essencialmente instável/próximo do acaso, o mesmo padrão de alerta visto no k-means e no
  k-medoids para k's com métricas isoladas boas mas instáveis.
- **k=5 e k=6 formam um segundo grupo competitivo em estabilidade** (0.438 e 0.443,
  respectivamente), com k=6 tendo o melhor Davies-Bouldin da tabela (1.727) — mas nenhum dos
  dois chega perto da estabilidade de k=3.
- **Candidatos selecionados para classificação: k=3 e k=6** — k=3 pela combinação isolada de
  estabilidade e Calinski-Harabasz mais fortes; k=6 como segundo candidato, pelo melhor
  Davies-Bouldin e estabilidade competitiva (levemente acima de k=5).

---

## 5. Árvore de Decisão + Grid Search (feature de cluster do K-Means)

### 5.1 Setup

- Base: `df_treino` completo (clientes que saíram e que não saíram), com a coluna `Saiu` como alvo.
- Features por teste: `variaveis_numericas` + `Cluster_k{k}` + `Distancia_Centroide_k{k}`
  (uma feature de cluster diferente para cada k testado, gerada a partir do k-means na seção 2).
- Split treino/teste único (80/20, `stratify=Saiu`, `random_state=42`), **reaproveitado em todos
  os k's**, para garantir que a comparação entre eles seja justa (só a feature de cluster muda).
- `GridSearchCV` sobre `DecisionTreeClassifier`, `cv=5`, `scoring='f1'` (accuracy não é uma boa
  métrica-guia aqui por causa do desbalanceamento das classes).
- Grid de hiperparâmetros: `max_depth` [3, 5, 7, 10, None], `min_samples_split` [2, 5, 10],
  `min_samples_leaf` [1, 2, 5], `criterion` ['gini', 'entropy'] — 90 combinações × 5 folds = 450
  treinos por valor de k.

### 5.2 Resultados

| k | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 2 | criterion=gini, max_depth=5, ... | 0.8386 | 0.6630 | 0.4211 | 0.5150 | 0.8199 |
| 3 | criterion=entropy, max_depth=5, ... | 0.8457 | 0.6806 | 0.4561 | **0.5462** | 0.8171 |
| 4 | criterion=entropy, max_depth=10, ... | 0.8057 | 0.5286 | 0.4211 | 0.4688 | 0.7618 |
| 5 | criterion=gini, max_depth=7, ... | 0.8314 | 0.6150 | 0.4596 | 0.5261 | 0.8150 |
| 6 | criterion=entropy, max_depth=10, ... | 0.8157 | 0.5534 | 0.4912 | 0.5204 | 0.7410 |
| 7 | criterion=gini, max_depth=7, ... | 0.8321 | 0.6289 | 0.4281 | 0.5094 | 0.8120 |

### 5.3 Interpretação

- **Melhor desempenho geral: k=3** — maior accuracy (0.846), maior precision (0.681) e maior F1
  (0.546); ROC-AUC (0.817) fica tecnicamente atrás só de k=2 (0.820), diferença desprezível.
- **Pior desempenho: k=4** — accuracy, precision, F1 e ROC-AUC mais baixos do grupo (junto com k=6).
- **k=6** tem o maior recall (0.491), ou seja, é o que mais identifica os clientes que de fato
  cancelaram — mas à custa de precision e ROC-AUC mais baixos, indicando pior generalização geral.
- **Recall baixo em todos os k's** (0.42–0.49): o modelo deixa passar mais da metade dos clientes
  que realmente cancelam, independente do k — ponto de atenção para negócio, possivelmente
  resolvido ajustando o threshold de decisão ou usando `class_weight='balanced'`.
- **Contradição relevante com a clusterização (seção 2):** k=4 havia se destacado nas métricas
  *internas* de clustering (estabilidade ARI de 0.87), mas aqui é o pior para prever `Saiu`. Já
  k=3, que era mediano na clusterização, é o melhor preditor. Isso reforça que "cluster bem
  formado" (separação/estabilidade) e "cluster útil para prever o alvo" são coisas diferentes —
  um cluster pode ser matematicamente consistente sem guardar relação com a variável de interesse.

### 5.4 Baseline (sem feature de cluster)

Mesmo setup da seção 5.1 (mesmo split treino/teste, mesma grid search), usando **apenas**
`variaveis_numericas` — sem `Cluster_k{k}` nem `Distancia_Centroide_k{k}`.

| | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline | criterion=gini, max_depth=7, ... | 0.8400 | 0.6473 | 0.4702 | 0.5447 | 0.8091 |

**Conclusão: a feature de cluster não agrega valor preditivo real.**

- O F1 do baseline (0.5447) é praticamente idêntico ao do melhor k, k=3 (0.5462) — diferença de
  0.0015, dentro da margem de ruído, não uma melhora real. Todos os demais k's (2, 4, 5, 6, 7)
  ficam **abaixo** do baseline em F1.
- O recall do baseline (0.4702) é maior que o de k=2, k=3, k=4, k=5 e k=7 — só perde para k=6
  (0.4912). Ou seja, adicionar a feature de cluster não ajudou a capturar mais casos de churn;
  em vários k's, atrapalhou.
- No ROC-AUC, o baseline (0.8091) fica no meio da tabela: perde para k=2, k=3, k=5, k=7, mas
  ganha de k=4 e k=6 — sem padrão claro de que o cluster ajuda ou atrapalha nessa métrica.
- **Interpretação prática:** as variáveis originais já carregam praticamente toda a informação
  que a árvore consegue usar para prever `Saiu`. A árvore de decisão, por natureza, já cria seus
  próprios "clusters implícitos" através dos splits — então a coluna de cluster gerada
  externamente via k-means é, na melhor das hipóteses (k=3), redundante, e na pior (k=4, k=6),
  uma variável de baixo sinal que confunde o modelo.

---

## 6. Random Forest + Grid Search (feature de cluster do K-Means)

### 6.1 Setup

- Mesmo split treino/teste (`idx_train`/`idx_test`) e mesmas features por k (`variaveis_numericas`
  + `Cluster_k{k}` + `Distancia_Centroide_k{k}`) usados na árvore de decisão (seção 5), para manter
  a comparação entre algoritmos justa.
- `GridSearchCV` sobre `RandomForestClassifier(random_state=42, n_jobs=-1)`, `cv=5`, `scoring='f1'`.
- Grid de hiperparâmetros: `n_estimators` [100, 200, 300], `max_depth` [5, 10, None],
  `min_samples_split` [2, 5], `min_samples_leaf` [1, 2], `max_features` ['sqrt', 'log2']
  — 72 combinações × 5 folds = 360 florestas treinadas por valor de k (`criterion` deixado de
  fora do grid para manter o custo computacional viável).

### 6.2 Resultados (com feature de cluster)

| k | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 2 | max_depth=10, max_features=log2, ... | 0.8407 | 0.6722 | 0.4246 | 0.5204 | 0.8212 |
| 3 | max_depth=None, max_features=log2, ... | 0.8350 | 0.6500 | 0.4105 | 0.5032 | 0.8174 |
| 4 | max_depth=None, max_features=log2, ... | 0.8314 | 0.6256 | 0.4281 | 0.5083 | 0.8171 |
| 5 | max_depth=None, max_features=log2, ... | 0.8293 | 0.6150 | 0.4316 | 0.5072 | 0.8187 |
| 6 | max_depth=None, max_features=log2, ... | 0.8307 | 0.6263 | 0.4175 | 0.5011 | 0.8185 |
| 7 | max_depth=10, max_features=log2, ... | 0.8393 | 0.6705 | 0.4140 | 0.5119 | **0.8267** |

### 6.3 Baseline (sem feature de cluster)

| | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline | max_depth=10, max_features=sqrt, ... | 0.8436 | 0.6919 | 0.4175 | **0.5208** | **0.8272** |

### 6.4 Interpretação

- **Variação entre k's muito menor que na árvore de decisão**: o F1 oscila só entre 0.501 e 0.520
  (range de 0.019), contra 0.469–0.546 (range de 0.077) na árvore isolada. Isso é esperado — o
  Random Forest combina centenas de árvores treinadas em subamostras com features sorteadas
  aleatoriamente, o que dilui o peso de qualquer feature individual, inclusive a de cluster.
- **O baseline vence (ou empata) em praticamente tudo**: F1 do baseline (0.5208) é o maior valor
  da tabela, levemente acima até de k=2 (0.5204, melhor com cluster); ROC-AUC do baseline (0.8272)
  também é o maior de todos, levemente acima de k=7 (0.8267); precision do baseline (0.6919)
  supera todos os k's. **Nenhuma versão com cluster supera o baseline de forma relevante.**
- **Confirma e reforça a conclusão da árvore de decisão (seção 5.4)**: a feature de cluster do
  k-means não agrega valor preditivo. Com Random Forest a evidência é ainda mais clara, já que
  aqui nem o melhor k conseguiu superar o baseline em nenhuma métrica.
- **Comparação entre algoritmos de classificação:**

  | | Melhor F1 | Melhor ROC-AUC |
  |---|---|---|
  | Árvore de decisão | 0.546 (k=3, com cluster) | 0.820 (k=2, com cluster) |
  | Random Forest | 0.521 (baseline) | 0.827 (baseline) |
  | XGBoost | 0.538 (k=4, com cluster) | 0.831 (k=4, com cluster) |

  A árvore de decisão (k=3) teve o melhor F1 entre árvore/RF. Já o XGBoost com k=4 foi o melhor
  resultado do estudo inteiro em ambas as métricas — e o único caso em que a feature de cluster
  superou claramente o baseline, coincidindo com o k que teve a maior estabilidade na
  clusterização (seção 2). O Random Forest baseline teve o melhor ROC-AUC entre árvore/RF.
- **Próximo passo sugerido:** extrair `feature_importances_` do melhor modelo de cada k para
  confirmar, numericamente, se `Cluster_k{k}` teve importância próxima de zero — mais uma
  evidência a favor da conclusão acima.

---

## 7. Comparação entre algoritmos

### 7.1 Clusterização — melhor k de cada algoritmo (métricas internas)

| Algoritmo | Melhor k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI) |
|---|---|---|---|---|---|
| **K-Means** | k=4 | 0.162 | 1.826 | **230.98** | **0.870** |
| Agglomerative (ward) | k=3 | 0.162 | 1.999 | 212.37 | 0.657 |
| K-Medoids | k=7 | 0.120 | 1.771 | 157.93 | 0.260 |

**Ranking de qualidade de clusterização: K-Means > Agglomerative > K-Medoids.** O k-means vence
com folga em estabilidade (a métrica mais decisiva) e em Calinski-Harabasz; o Agglomerative fica
em segundo lugar competitivo (estabilidade de 0.657 não é desprezível); o K-Medoids é claramente
o mais fraco dos três em praticamente todas as métricas.

### 7.2 Ressalva importante — clusterização ≠ poder preditivo

Como já vimos nas seções 5 e 6, "melhor clusterização" (métricas internas) e "melhor feature para
prever `Saiu`" (métricas de classificação) já se mostraram coisas diferentes: o k=4 do k-means,
vencedor isolado em estabilidade, foi o pior para árvore de decisão e Random Forest, e só se
revelou útil no XGBoost. Por isso, o ranking acima **não define sozinho** qual algoritmo/k deve
seguir para a etapa de classificação — ele serve para justificar a escolha dos candidatos
(k-means k=3/k=4, Agglomerative k=3/k=6, K-Medoids k=6/k=7), mas a palavra final depende de como
cada um se sai nos classificadores (árvore, Random Forest, XGBoost, SVM, Naive Bayes + baseline).

*(preencher, após testar os candidatos de Agglomerative e K-Medoids nos classificadores: qual
combinação algoritmo + k + classificador deu o melhor resultado preditivo, e se as segmentações
fazem sentido de negócio ao olhar o perfil médio de cada cluster nas variáveis originais)*

---

## 8. Conclusão e recomendação final

*(preencher ao final — recomendação de algoritmo/k para seguir para a etapa de classificação)*
