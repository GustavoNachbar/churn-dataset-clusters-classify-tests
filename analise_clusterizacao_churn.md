# Análise de Clusterização — Segmentação de Clientes que Cancelaram (Churn)

**Objetivo:** testar diferentes algoritmos de clusterização (k de 2 a 7) sobre os clientes que
cancelaram (`Saiu == 1`), usando as variáveis `PontuacaoCredito`, `Idade`, `TempoRelacionamento`,
`Saldo`, `NumeroProdutos` e `SalarioEstimado` (padronizadas), para identificar perfis distintos
de churn e validar qual número de clusters melhor representa a estrutura dos dados.

**Estrutura do documento:** clusterização (algoritmos → métricas) → classificação (algoritmos →
métricas) → conclusão. Cada bloco de "algoritmos" traz o setup/código; cada bloco de "métricas"
traz resultados, interpretação e comparações.

---

## 1. Glossário das métricas de clusterização

| Métrica | O que mede | Direção ideal | Observações |
|---|---|---|---|
| **Inertia** | Soma das distâncias quadráticas de cada ponto ao centróide do seu cluster | Quanto menor, melhor (mas sempre cai com mais clusters) | Usada só para o "método do cotovelo" — não comparar valor absoluto entre algoritmos diferentes |
| **Silhouette Score** | Quão bem separado e coeso cada ponto está em relação ao seu cluster vs. outros clusters | Quanto maior, melhor (varia de -1 a 1) | > 0.5 = estrutura forte; 0.25–0.5 = estrutura fraca/moderada; < 0.25 = pouca separação real |
| **Davies-Bouldin Index** | Razão entre dispersão interna dos clusters e distância entre eles | Quanto menor, melhor | Sensível a clusters de tamanhos/densidades muito diferentes |
| **Calinski-Harabasz Index** | Razão entre dispersão entre clusters e dispersão dentro dos clusters | Quanto maior, melhor | Tende a favorecer clusters convexos e de tamanho parecido |
| **Estabilidade (ARI médio via bootstrap)** | Quão consistente é o agrupamento ao re-rodar o algoritmo em subamostras dos dados | Quanto mais perto de 1, melhor | Desvio-padrão alto = solução instável, sensível a qual subconjunto de dados é usado |

---

## 2. Algoritmos de Clusterização (setup)

### 2.1 K-Means

- Pré-processamento: `StandardScaler` nas 6 variáveis numéricas, ajustado em `df_treino_cancelamento`
- `KMeans(n_init=10, random_state=42)`, testado para k = 2 a 8
- Estabilidade: bootstrap com 30 reamostragens, 80% dos dados por rodada, comparação via Adjusted Rand Index (ARI) contra o clustering na base completa

### 2.2 K-Medoids

- Mesma base (`df_treino_cancelamento`) e mesmo `StandardScaler` (ajustado separadamente para
  essa base, não reaproveitado do `df_treino` completo).
- `KMedoids(method='alternate', init='k-medoids++', random_state=42)`, testado para k = 2 a 7.
- Estabilidade calculada com a mesma lógica de bootstrap + ARI usada no k-means.

### 2.3 Agglomerative Clustering (Clustering Hierárquico)

- Mesma base (`df_treino_cancelamento`) e mesmo `StandardScaler`.
- `AgglomerativeClustering(linkage='ward')`, testado para k = 2 a 7.
- Estabilidade calculada com a mesma lógica de bootstrap + ARI (sem `random_state`, já que o
  algoritmo é determinístico dado o dataset). Diferente do K-Means e K-Medoids, não possui
  `.predict()` nem `.transform()` — não há coluna de distância ao centro para esse algoritmo, e
  o `fit_predict()` precisa rodar direto na base completa (sem "transferir" um modelo já treinado).

---

## 3. Métricas de Clusterização (resultados e interpretação)

### 3.1 K-Means

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|---|
| 2 | 7368.29 | 0.1723 | 2.1604 | 229.54 | 0.6414 | 0.4466 |
| 3 | 6483.43 | 0.1781 | 1.8807 | 227.45 | 0.5361 | 0.2889 |
| 4 | 5752.68 | 0.1620 | 1.8257 | 230.98 | **0.8696** | **0.1211** |
| 5 | 5343.00 | 0.1430 | 1.6733 | 213.64 | 0.5981 | 0.0818 |
| 6 | 5024.91 | 0.1419 | 1.5865 | 199.57 | 0.5689 | 0.1151 |
| 7 | 4768.35 | 0.1439 | 1.5780 | 187.86 | 0.5082 | 0.0780 |
| 8 | 4526.41 | 0.1453 | 1.5987 | 180.34 | 0.5151 | 0.0868 |

**Interpretação:**
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
- **Conclusão parcial:** k=3 não é claramente ruim, mas também não se destaca — vence o
  silhouette por margem pequena e perde nas demais métricas. **k=4** é o candidato mais forte,
  por reunir estabilidade muito superior e métricas de separação comparáveis ou melhores.
- **Candidatos selecionados para classificação: k=3 e k=4.**
- **Ressalva geral:** como o silhouette é baixo mesmo no melhor k, vale considerar, em versões
  futuras, incluir variáveis adicionais (categóricas/comportamentais) ou testar algoritmos que
  lidam melhor com fronteiras pouco nítidas entre grupos.

### 3.2 K-Medoids

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|
| 2 | 0.1174 | 2.6777 | 184.02 | 0.0655 | 0.0806 |
| 3 | 0.0971 | 2.3613 | 159.32 | 0.1235 | 0.0668 |
| 4 | 0.1029 | 2.0737 | 160.24 | 0.1622 | 0.0501 |
| 5 | 0.1159 | 1.8890 | 164.82 | 0.2337 | 0.0650 |
| 6 | 0.1091 | 1.7759 | 158.50 | **0.2615** | 0.0519 |
| 7 | **0.1200** | **1.7710** | 157.93 | 0.2603 | 0.0554 |

**Interpretação:**
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

**Métricas na base geral (`df_treino` completo)** — recalculadas reaproveitando os labels já
gerados em `Cluster_k6`/`Cluster_k7` no `df_treino` completo (7.000 clientes, cancelados e não
cancelados), com `X_scaled_geral` próprio dessa base:

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|
| 6 | 0.1323 | 1.8744 | 804.23 | 0.2859 | 0.0571 |
| 7 | 0.1306 | 1.8300 | 772.90 | 0.2694 | 0.0538 |

- Padrão qualitativamente igual ao da base de cancelados: separação fraca (silhouette baixo) e
  estabilidade modesta, bem abaixo do k-means. k=6 continua levemente à frente de k=7 (estabilidade
  0.286 vs 0.269), mesma hierarquia da base de cancelados.
- Calinski-Harabasz não é comparável entre as duas bases (cresce com o tamanho da amostra: ~7.000
  vs ~1.426 clientes) — a diferença de escala não indica melhora real de separação.
- Não há motivo, com esses números, para reconsiderar os candidatos já escolhidos (k=6 e k=7).

### 3.3 Agglomerative Clustering

| k | Linkage | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|---|
| 2 | ward | 0.1638 | 2.1841 | 191.81 | 0.1598 | 0.2372 |
| 3 | ward | 0.1616 | 1.9988 | **212.37** | **0.6574** | 0.1302 |
| 4 | ward | 0.1189 | 2.0308 | 192.90 | 0.4227 | 0.0797 |
| 5 | ward | 0.1018 | 1.8302 | 177.07 | 0.4385 | 0.0687 |
| 6 | ward | 0.0909 | **1.7268** | 160.20 | 0.4431 | 0.0648 |
| 7 | ward | 0.0967 | 1.8473 | 149.76 | 0.3797 | 0.0576 |

**Interpretação:**
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

**Métricas na base geral (`df_treino` completo)** — recalculadas reaproveitando os labels já
gerados em `Cluster_k3`/`Cluster_k6` no `df_treino` completo, com `X_scaled_geral` próprio dessa
base:

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI médio) | Estabilidade (desvio) |
|---|---|---|---|---|---|
| 3 | **0.1502** | 2.1211 | 1023.66 | **0.7002** | 0.1696 |
| 6 | 0.1051 | **1.8670** | 772.37 | 0.5631 | 0.0915 |

- **k=3 continua vencendo k=6 com folga**, agora com vantagem ainda maior em estabilidade (0.700
  vs 0.563) do que na base de cancelados (0.657 vs 0.443). Davies-Bouldin favorece levemente k=6,
  mas não compensa a diferença em estabilidade e silhouette a favor de k=3.
- Calinski-Harabasz não é comparável entre as duas bases (mesmo efeito de escala do K-Medoids).
- **Comparando com o K-Medoids na base geral**: o Agglomerative k=3 (estabilidade 0.700) é
  disparado mais forte que o melhor resultado do K-Medoids (k=6, estabilidade 0.286) — mais que
  o dobro. Entre os dois "desafiantes" do k-means, o **Agglomerative k=3** é o candidato mais
  promissor para competir na etapa de classificação.

### 3.4 Ranking geral de todos os k's testados (score composto)

**Metodologia**: cada métrica foi normalizada (min-max) entre os 19 resultados (7 do K-Means,
6 do Agglomerative, 6 do K-Medoids, todos calculados em `df_treino_cancelamento`); Davies-Bouldin
foi invertido antes de normalizar (menor valor bruto = melhor). Score final = 40% estabilidade +
20% silhouette + 20% Davies-Bouldin + 20% Calinski-Harabasz — peso maior na estabilidade por ser,
empiricamente, a métrica mais decisiva neste estudo.

| # | Algoritmo | k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade | Score | Confiabilidade |
|---|---|---|---|---|---|---|---|---|
| 1 | K-Means | 4 | 0.162 | 1.826 | 230.98 | 0.870 | **0.918** | OK |
| 2 | K-Means | 3 | 0.178 | 1.881 | 227.45 | 0.536 | 0.770 | OK |
| 3 | K-Means | 2 | 0.172 | 2.160 | 229.54 | 0.641 | 0.764 | ⚠️ desvio elevado (0.447) |
| 4 | Agglomerative | 3 | 0.162 | 1.999 | 212.37 | 0.657 | 0.734 | OK |
| 5 | K-Means | 5 | 0.143 | 1.673 | 213.64 | 0.598 | 0.724 | OK |
| 6 | K-Means | 6 | 0.142 | 1.587 | 199.57 | 0.569 | 0.689 | OK |
| 7 | K-Means | 7 | 0.144 | 1.578 | 187.86 | 0.508 | 0.636 | OK |
| 8 | K-Means | 8 | 0.145 | 1.599 | 180.34 | 0.515 | 0.620 | OK |
| 9 | Agglomerative | 4 | 0.119 | 2.031 | 192.90 | 0.423 | 0.466 | OK |
| 10 | Agglomerative | 5 | 0.102 | 1.830 | 177.07 | 0.439 | 0.432 | OK |
| 11 | Agglomerative | 2 | 0.164 | 2.184 | 191.81 | 0.160 | 0.407 | 🚫 desvio (0.237) > média |
| 12 | Agglomerative | 6 | 0.091 | 1.727 | 160.20 | 0.443 | 0.386 | OK |
| 13 | K-Medoids | 7 | 0.120 | 1.771 | 157.93 | 0.260 | 0.349 | OK |
| 14 | K-Medoids | 6 | 0.109 | 1.776 | 158.50 | 0.262 | 0.325 | OK |
| 15 | K-Medoids | 5 | 0.116 | 1.889 | 164.82 | 0.234 | 0.322 | OK |
| 16 | Agglomerative | 7 | 0.097 | 1.847 | 149.76 | 0.380 | 0.321 | OK |
| 17 | K-Medoids | 4 | 0.103 | 2.074 | 160.24 | 0.162 | 0.211 | OK |
| 18 | K-Medoids | 2 | 0.117 | 2.678 | 184.02 | 0.065 | 0.145 | 🚫 desvio (0.081) > média |
| 19 | K-Medoids | 3 | 0.097 | 2.361 | 159.32 | 0.124 | 0.124 | OK |

**Leituras principais:**
- **K-Means k=4 confirma-se como o melhor de todos** os 19 candidatos, com folga expressiva para
  o segundo colocado.
- **Os 6 primeiros lugares são todos do K-Means** — reforça que ele é, disparado, o algoritmo
  mais forte de clusterização para este dataset.
- **O melhor resultado do Agglomerative (k=3) fica em 4º lugar geral**, à frente de vários k's
  do próprio K-Means (5, 6, 7, 8).
- **O K-Medoids nunca aparece antes da 13ª posição** — confirma que é o mais fraco dos três
  algoritmos em qualquer recorte.
- Linhas marcadas com 🚫 (Agglomerative k=2 e K-Medoids k=2) têm desvio-padrão de estabilidade
  maior que a própria média — resultado próximo do acaso, apesar de entrarem bem posicionadas
  no score composto; tratar com desconfiança mesmo com score aparentemente competitivo.

### 3.5 Comparação entre algoritmos (melhor k de cada um)

| Algoritmo | Melhor k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Estabilidade (ARI) |
|---|---|---|---|---|---|
| **K-Means** | k=4 | 0.162 | 1.826 | **230.98** | **0.870** |
| Agglomerative (ward) | k=3 | 0.162 | 1.999 | 212.37 | 0.657 |
| K-Medoids | k=7 | 0.120 | 1.771 | 157.93 | 0.260 |

**Ranking de qualidade de clusterização: K-Means > Agglomerative > K-Medoids.** O k-means vence
com folga em estabilidade (a métrica mais decisiva) e em Calinski-Harabasz; o Agglomerative fica
em segundo lugar competitivo (estabilidade de 0.657 não é desprezível); o K-Medoids é claramente
o mais fraco dos três em praticamente todas as métricas.

### 3.6 Ressalva importante — clusterização ≠ poder preditivo

"Melhor clusterização" (métricas internas) e "melhor feature para prever `Saiu`" (métricas de
classificação) já se mostraram coisas diferentes: o k=4 do k-means, vencedor isolado em
estabilidade, foi o pior para árvore de decisão e Random Forest, e só se revelou útil no XGBoost
(seção 5). Por isso, o ranking acima **não define sozinho** qual algoritmo/k deve seguir para a
etapa de classificação — ele serve para justificar a escolha dos candidatos (k-means k=3/k=4,
Agglomerative k=3/k=6, K-Medoids k=6/k=7), mas a palavra final depende de como cada um se sai nos
classificadores (árvore, Random Forest, XGBoost, SVM, Naive Bayes + baseline).

*(preencher, após testar os candidatos de Agglomerative e K-Medoids nos classificadores: qual
combinação algoritmo + k + classificador deu o melhor resultado preditivo, e se as segmentações
fazem sentido de negócio ao olhar o perfil médio de cada cluster nas variáveis originais)*

---

## 4. Algoritmos de Classificação (setup)

### 4.1 Árvore de Decisão + Grid Search (feature de cluster do K-Means)

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

### 4.2 Random Forest + Grid Search (feature de cluster do K-Means)

- Mesmo split treino/teste (`idx_train`/`idx_test`) e mesmas features por k (`variaveis_numericas`
  + `Cluster_k{k}` + `Distancia_Centroide_k{k}`) usados na árvore de decisão, para manter a
  comparação entre algoritmos justa.
- `GridSearchCV` sobre `RandomForestClassifier(random_state=42, n_jobs=-1)`, `cv=5`, `scoring='f1'`.
- Grid de hiperparâmetros: `n_estimators` [100, 200, 300], `max_depth` [5, 10, None],
  `min_samples_split` [2, 5], `min_samples_leaf` [1, 2], `max_features` ['sqrt', 'log2']
  — 72 combinações × 5 folds = 360 florestas treinadas por valor de k (`criterion` deixado de
  fora do grid para manter o custo computacional viável).

---

## 5. Métricas de Classificação (resultados e interpretação)

### 5.1 Árvore de Decisão

| k | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 2 | criterion=gini, max_depth=5, ... | 0.8386 | 0.6630 | 0.4211 | 0.5150 | 0.8199 |
| 3 | criterion=entropy, max_depth=5, ... | 0.8457 | 0.6806 | 0.4561 | **0.5462** | 0.8171 |
| 4 | criterion=entropy, max_depth=10, ... | 0.8057 | 0.5286 | 0.4211 | 0.4688 | 0.7618 |
| 5 | criterion=gini, max_depth=7, ... | 0.8314 | 0.6150 | 0.4596 | 0.5261 | 0.8150 |
| 6 | criterion=entropy, max_depth=10, ... | 0.8157 | 0.5534 | 0.4912 | 0.5204 | 0.7410 |
| 7 | criterion=gini, max_depth=7, ... | 0.8321 | 0.6289 | 0.4281 | 0.5094 | 0.8120 |

**Baseline (sem feature de cluster)** — mesmo split treino/teste, mesma grid search, usando
**apenas** `variaveis_numericas`:

| | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline | criterion=gini, max_depth=7, ... | 0.8400 | 0.6473 | 0.4702 | 0.5447 | 0.8091 |

**Interpretação:**
- **Melhor desempenho geral: k=3** — maior accuracy (0.846), maior precision (0.681) e maior F1
  (0.546); ROC-AUC (0.817) fica tecnicamente atrás só de k=2 (0.820), diferença desprezível.
- **Pior desempenho: k=4** — accuracy, precision, F1 e ROC-AUC mais baixos do grupo (junto com k=6).
- **k=6** tem o maior recall (0.491), ou seja, é o que mais identifica os clientes que de fato
  cancelaram — mas à custa de precision e ROC-AUC mais baixos, indicando pior generalização geral.
- **Recall baixo em todos os k's** (0.42–0.49): o modelo deixa passar mais da metade dos clientes
  que realmente cancelam, independente do k — ponto de atenção para negócio, possivelmente
  resolvido ajustando o threshold de decisão ou usando `class_weight='balanced'`.
- **Contradição relevante com a clusterização (seção 3.1):** k=4 havia se destacado nas métricas
  *internas* de clustering (estabilidade ARI de 0.87), mas aqui é o pior para prever `Saiu`. Já
  k=3, que era mediano na clusterização, é o melhor preditor. Isso reforça que "cluster bem
  formado" (separação/estabilidade) e "cluster útil para prever o alvo" são coisas diferentes.
- **Baseline vs. cluster:** o F1 do baseline (0.5447) é praticamente idêntico ao do melhor k, k=3
  (0.5462) — diferença de 0.0015, dentro da margem de ruído. Todos os demais k's ficam abaixo do
  baseline em F1. O recall do baseline (0.4702) é maior que o de quase todos os k's (só perde
  para k=6). **Conclusão: a feature de cluster do k-means não agrega valor preditivo real** para
  a árvore de decisão — as variáveis originais já carregam praticamente toda a informação que o
  modelo consegue usar; a árvore já cria seus próprios "clusters implícitos" via splits.

### 5.2 Random Forest

**Resultados (com feature de cluster):**

| k | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 2 | max_depth=10, max_features=log2, ... | 0.8407 | 0.6722 | 0.4246 | 0.5204 | 0.8212 |
| 3 | max_depth=None, max_features=log2, ... | 0.8350 | 0.6500 | 0.4105 | 0.5032 | 0.8174 |
| 4 | max_depth=None, max_features=log2, ... | 0.8314 | 0.6256 | 0.4281 | 0.5083 | 0.8171 |
| 5 | max_depth=None, max_features=log2, ... | 0.8293 | 0.6150 | 0.4316 | 0.5072 | 0.8187 |
| 6 | max_depth=None, max_features=log2, ... | 0.8307 | 0.6263 | 0.4175 | 0.5011 | 0.8185 |
| 7 | max_depth=10, max_features=log2, ... | 0.8393 | 0.6705 | 0.4140 | 0.5119 | **0.8267** |

**Baseline (sem feature de cluster):**

| | Melhores parâmetros | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline | max_depth=10, max_features=sqrt, ... | 0.8436 | 0.6919 | 0.4175 | **0.5208** | **0.8272** |

**Interpretação:**
- **Variação entre k's muito menor que na árvore de decisão**: o F1 oscila só entre 0.501 e 0.520
  (range de 0.019), contra 0.469–0.546 (range de 0.077) na árvore isolada. Isso é esperado — o
  Random Forest combina centenas de árvores treinadas em subamostras com features sorteadas
  aleatoriamente, o que dilui o peso de qualquer feature individual, inclusive a de cluster.
- **O baseline vence (ou empata) em praticamente tudo**: F1 do baseline (0.5208) é o maior valor
  da tabela, levemente acima até de k=2 (0.5204, melhor com cluster); ROC-AUC do baseline (0.8272)
  também é o maior de todos, levemente acima de k=7 (0.8267); precision do baseline (0.6919)
  supera todos os k's. **Nenhuma versão com cluster supera o baseline de forma relevante.**
- **Confirma e reforça a conclusão da árvore de decisão**: a feature de cluster do k-means não
  agrega valor preditivo. Com Random Forest a evidência é ainda mais clara, já que aqui nem o
  melhor k conseguiu superar o baseline em nenhuma métrica.

**Comparação entre algoritmos de classificação (melhores resultados de cada um):**

| | Melhor F1 | Melhor ROC-AUC |
|---|---|---|
| Árvore de decisão | 0.546 (k=3, com cluster) | 0.820 (k=2, com cluster) |
| Random Forest | 0.521 (baseline) | 0.827 (baseline) |
| XGBoost | 0.538 (k=4, com cluster) | 0.831 (k=4, com cluster) |

A árvore de decisão (k=3) teve o melhor F1 entre árvore/RF. Já o XGBoost com k=4 foi o melhor
resultado do estudo inteiro em ambas as métricas — e o único caso em que a feature de cluster
superou claramente o baseline, coincidindo com o k que teve a maior estabilidade na clusterização
(seção 3.1). O Random Forest baseline teve o melhor ROC-AUC entre árvore/RF.

**Próximo passo sugerido:** extrair `feature_importances_` do melhor modelo de cada k para
confirmar, numericamente, se `Cluster_k{k}` teve importância próxima de zero — mais uma evidência
a favor da conclusão acima.

---

## 6. Conclusão e recomendação final

### 6.1 O processo de decisão em 4 camadas

A escolha do "melhor cluster" não se apoiou numa métrica isolada — foi a convergência de quatro
checagens independentes, cada uma respondendo a uma pergunta diferente:

| Camada | Pergunta | Resultado |
|---|---|---|
| **1. Métricas de clusterização** | Qual agrupamento é matematicamente mais nítido e consistente? | **K-Means k=4** vence com folga (score composto 0.918, seção 3.4) |
| **2. Validação de face (perfil)** | Os grupos que a métrica elogiou fazem sentido em português? | Sim — o k=4 revelou uma divisão real dentro do grupo majoritário do k=3: "cliente recente, salário menor" (30,3% da base) vs. "cliente antigo, salário maior" (33,0% da base), diferenciados por `TempoRelacionamento` e `SalarioEstimado` |
| **3. Representatividade** | Os grupos são balanceados e evitam isolar outliers? | Sim — o menor grupo do k=4 tem 14,17% da base (202 clientes), nenhum cluster é ínfimo |
| **4. Poder preditivo** | O cluster ajuda de fato a prever `Saiu`? | Sim, mas só no XGBoost — k=4 foi o único caso em todo o estudo em que a feature de cluster superou o baseline (F1 de 0.538 e ROC-AUC de 0.831, seção 5.2) |

### 6.2 Recomendação final

**K-Means com k=4** é a recomendação para seguir à etapa de validação, pelas seguintes razões,
em ordem de peso:

1. É o único candidato, entre os 19 testados (seção 3.4), que passou nas quatro camadas de
   validação simultaneamente — as demais opções fortes (Agglomerative k=3, K-Means k=3) falharam
   na camada 4: nenhuma delas superou o baseline em nenhum classificador testado.
2. O ganho preditivo é modesto (cerca de 1 ponto percentual de F1 sobre o baseline), então a
   recomendação prática é usar `Cluster_k4` como feature **especificamente em conjunto com
   XGBoost** — não há evidência de que ela ajude árvore de decisão, Random Forest, ou (ainda a
   confirmar) SVM/Naive Bayes.
3. O perfil interpretável do k=4 (diferenciação por tempo de relacionamento e salário) tem valor
   de negócio mesmo além da classificação — pode orientar ações de retenção segmentadas por
   "cliente novo" vs. "cliente fidelizado" entre quem tem maior propensão a cancelar.

### 6.3 Ressalvas e próximos passos

- Os candidatos do Agglomerative (k=3, k=6) e K-Medoids (k=6, k=7) ainda não foram testados nos
  classificadores — a recomendação acima pode mudar se algum desses, mesmo com métricas de
  clusterização mais fracas, repetir o padrão do k=4 (surpresa positiva num classificador
  específico). Vale rodar essa bateria antes de fechar a conclusão de forma definitiva.
- Ainda não foram gerados os perfis dos candidatos do Agglomerative e K-Medoids — recomendável
  completar essa análise para os 4 candidatos restantes, seguindo o mesmo processo de 4 camadas.
- O silhouette baixo em todos os 19 candidatos testados (máximo de 0.178) é uma limitação
  estrutural do estudo: mesmo o "melhor" cluster não representa uma separação forte entre os
  clientes. Isso não invalida o k=4 como escolha relativa, mas reforça que a segmentação por
  essas 6 variáveis numéricas tem valor limitado — incluir variáveis categóricas/comportamentais
  em trabalhos futuros pode melhorar a qualidade da segmentação de forma mais substancial.
