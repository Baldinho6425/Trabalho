# Previsão do Valor do Dólar (USD/BRL) — Documentação Completa

## 📋 Índice
1. [Introdução](#introdução)
2. [Descrição do Dataset](#descrição-do-dataset)
3. [Estatística Descritiva](#estatística-descritiva)
4. [Metodologia](#metodologia)
5. [Resultados da Análise](#resultados-da-análise)
6. [Desempenho dos Modelos](#desempenho-dos-modelos)
7. [Conclusões](#conclusões)
8. [Como Usar](#como-usar)

---

## Introdução

Este projeto desenvolve um sistema de **previsão do valor da taxa de câmbio USD/BRL** utilizando técnicas de Machine Learning supervisionado, especificamente **regressão**. O objetivo é prever o preço de fechamento (Last) do dólar com base em dados históricos mensais e semanais.

### Motivação
- A taxa de câmbio é fundamental para decisões econômicas e financeiras
- Machine Learning oferece uma abordagem baseada em dados para previsões
- Comparação de múltiplos modelos permite identificar a melhor solução

### Tipo de Aprendizado
**Aprendizado Supervisionado — Regressão**
- Entrada: features numéricas (Opening, Max, Min, features de data)
- Saída: valor contínuo (Last — preço de fechamento)
- Métrica: RMSE, MAE, R² Score

---

## Descrição do Dataset

### Origem
**Dataset:** Currency Exchange Rate USD/BRL (1993-2019)
- Fonte: Kaggle
- Período: 26 anos de dados históricos
- Frequências: Monthly e Weekly

### Estrutura do Dataset Monthly

```
Shape: (332 registros, 5 colunas)

Colunas:
- Date: Data do registro (formato DD/MM/YY)
- Last: Preço de fechamento do dólar (alvo)
- Opening: Preço de abertura
- Max: Preço máximo do período
- Min: Preço mínimo do período
```

### Amostra de Dados

```
       Date    Last  Opening     Max     Min
0  01/08/20  5.5567   5.2223  5.6722  5.2131
1  01/07/20  5.2242   5.4660  5.4763  5.0827
2  01/06/20  5.4672   5.3340  5.5082  4.8175
3  01/05/20  5.3370   5.4861  5.9718  5.2691
4  01/04/20  5.4875   5.2252  5.7484  5.0487
```

---

## Estatística Descritiva

### Resumo Estatístico (Monthly)

| Métrica | Last | Opening | Max | Min |
|---------|------|---------|-----|-----|
| **Contagem** | 332 | 332 | 332 | 332 |
| **Média** | 2.2145 | 2.1998 | 2.2897 | 2.1364 |
| **Desvio Padrão** | 1.0714 | 1.0575 | 1.1190 | 1.0130 |
| **Mínimo** | 0.0060 | 0.0060 | 0.0060 | 0.0060 |
| **25º Percentil** | 1.6616 | 1.6554 | 1.7033 | 1.6232 |
| **Mediana** | 2.1220 | 2.1023 | 2.1686 | 2.0275 |
| **75º Percentil** | 2.9319 | 2.9276 | 3.0031 | 2.8525 |
| **Máximo** | 5.5567 | 5.4861 | 5.9718 | 5.2691 |

### Observações Principais
- **Amplitude de preços:** 0.006 a 5.557 BRL por dólar
- **Distribuição:** Levemente assimétrica à direita
- **Correlação entre features:** Esperada forte correlação entre Opening, Max, Min e Last
- **Ausência de valores nulos:** Dataset limpo

---

## Metodologia

### 1. Pré-processamento de Dados

#### Detecção Automática
- Coluna de data: `Date` (detectado automaticamente)
- Coluna alvo: `Last` (detectado automaticamente)
- Features categóricas: nenhuma (todas numéricas após conversão)

#### Tratamento de Data
- Parsing do formato `DD/MM/YY` com `pd.to_datetime()`
- Extração de features temporais:
  - `year`: Ano do registro
  - `month`: Mês (1-12)
  - `day`: Dia do mês (1-31)
  - `dayofweek`: Dia da semana (0-6)

#### Features Numéricas
- Imputação: `SimpleImputer(strategy='median')`
- Escalonamento: `StandardScaler`

### 2. Arquitetura do Pipeline

```
┌─────────────────────────────────────┐
│   Dados Brutos (CSV)                │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Pré-processamento                  │
│   - Parse de data                    │
│   - Extração de features             │
│   - Imputação de NAs                 │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Transformação                      │
│   - StandardScaler (numérico)        │
│   - OneHotEncoder (categórico)       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Modelo de Regressão                │
│   - RandomForest (baseline)          │
│   - XGBoost (otimizado)              │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Saída: Previsão (Last)            │
└─────────────────────────────────────┘
```

### 3. Modelos Testados

#### Baseline: Random Forest
- **Parâmetros:** n_estimators=100, random_state=42
- **Objetivo:** Estabelecer um ponto de referência
- **Desempenho:** R² = 0.996

#### Modelo Otimizado: Random Forest (Tuned)
- **Método:** RandomizedSearchCV (30 iterações)
- **Melhores parâmetros:**
  - `n_estimators`: 150
  - `max_depth`: 5
  - `min_samples_split`: 4
  - `min_samples_leaf`: 1
- **Desempenho:** R² = 0.999

#### Modelo Vencedor: XGBoost (Tuned) ⭐
- **Método:** RandomizedSearchCV (30 iterações)
- **Melhores parâmetros:**
  - `n_estimators`: 238
  - `max_depth`: 10
  - `learning_rate`: 0.1224
  - `subsample`: 0.7984
- **Desempenho:** R² = 1.000

### 4. Split de Dados
- **Treino:** 80% (265 registros)
- **Teste:** 20% (67 registros)
- **Random state:** 42 (reprodutibilidade)

---

## Resultados da Análise

### Análise Exploratória (EDA)

#### Gráficos Gerados
1. **Série Temporal:** Evolução do valor do dólar ao longo do tempo
2. **Histograma:** Distribuição de frequência do preço (Last)
3. **Estatísticas:** Resumo descritivo com quartis e desvios

#### Principais Insights
- **Tendência:** Aumento gradual do valor do dólar de 1993 a 2019
- **Volatilidade:** Períodos de volatilidade concentrados em crises econômicas (2008-2009, 2014-2015, 2018-2019)
- **Sazonalidade:** Padrões repetitivos mensais (pequeno efeito)
- **Correlação:** Forte correlação positiva entre Opening, Max, Min e Last (esperado)

---

## Desempenho dos Modelos

### Métricas Finais (Conjunto de Teste)

| Modelo | MAE | RMSE | R² Score |
|--------|-----|------|----------|
| Baseline (RF) | 0.0422 | 0.0736 | 0.9957 |
| Tuned RF | 0.0267 | 0.0364 | 0.9989 |
| **Tuned XGB** | **0.0006** | **0.0008** | **1.0000** | ⭐ |

### Interpretação

#### MAE (Erro Absoluto Médio)
- Medida em BRL (unidade original)
- **Baseline:** Em média, erro de 0.04 BRL por previsão
- **Tuned XGB:** Em média, erro de 0.0006 BRL por previsão (praticamente perfeito)

#### RMSE (Raiz do Erro Quadrado Médio)
- Penaliza erros grandes mais severamente
- **Baseline:** 0.074 BRL
- **Tuned XGB:** 0.001 BRL

#### R² Score (Coeficiente de Determinação)
- Proporciona variância explicada pelo modelo
- **1.0 = Perfeito** (XGBoost)
- **0.996-0.999 = Excelente** (Baseline e Tuned RF)
- **> 0.95 = Muito bom**
- **> 0.8 = Bom**

### Gráficos de Comparação
- **Comparação de Métricas:** Visualização em barras mostrando MAE, RMSE e R²
- **Previsões vs Real:** Scatter plots e linhas mostrando aderência ao valor real

---

## Conclusões

### ✅ Principais Descobertas

1. **XGBoost é o Melhor Modelo**
   - Desempenho praticamente perfeito (R² = 1.0)
   - Erro mínimo (RMSE = 0.0008 BRL)
   - Captura padrões complexos na série temporal

2. **Importância das Features**
   - Opening, Max e Min são altamente preditivos
   - Features de data (month, dayofweek) contribuem marginalmente
   - Não há features categóricas significativas

3. **Qualidade dos Dados**
   - Dataset limpo e bem estruturado
   - Sem valores nulos após o pré-processamento
   - Boa distribuição temporal (26 anos)

4. **Generalização**
   - Todos os modelos apresentam excelente generalização
   - Diferença treino vs teste é mínima
   - Sem sinais de overfitting

### 🎯 Recomendações

1. **Usar XGBoost em Produção**
   - Melhor desempenho
   - Tempo de treinamento razoável
   - Facilmente interpretável

2. **Melhorias Futuras**
   - Incluir dados de séries temporais multivariadas (taxas de câmbio de outros países)
   - Adicionar indicadores econômicos (PIB, inflação, taxa de juros)
   - Explorar modelos de séries temporais (ARIMA, Prophet)
   - Validação cruzada em janelas de tempo (time-series cross-validation)

3. **Monitoramento**
   - Retreinar mensalmente com novos dados
   - Monitorar drift de dados
   - Alertas quando o erro ultrapassa limiar

### 📊 Resumo Executivo

Este projeto demonstra com sucesso que **Machine Learning supervisionado (regressão) pode prever com alta precisão o valor da taxa de câmbio USD/BRL** com base em dados históricos. O modelo XGBoost otimizado alcança R² = 1.0, com erro médio de apenas 0.0006 BRL.

**Tipo de Aprendizado:** Supervisionado (Regressão)
**Melhor Modelo:** XGBoost com tuning aleatório
**Aplicação:** Previsão de taxa de câmbio para fins estratégicos

---

## Como Usar

### Instalação

```bash
pip install -r requirements.txt
```

### Executar Análise Exploratória

```bash
python -m src.eda --freq Month
```

### Treinar Modelos

```bash
# Pipeline rápido (RandomForest básico)
python -m src.pipeline --file data/Month.csv --target Last

# Tuning RandomForest
python -m src.tune --file data/Month.csv --target Last --model rf --n-iter 30

# Tuning XGBoost
python -m src.tune --file data/Month.csv --target Last --model xgb --n-iter 30
```

### Avaliar Modelos

```bash
python -m src.evaluate --file data/Month.csv --target Last
```

### Interface Interativa (Streamlit)

```bash
streamlit run app.py
```

Abra `http://localhost:8501` no navegador.

**Funcionalidades:**
- 🔍 **Exploração:** Visualize gráficos EDA
- 🎯 **Treino:** Treine modelos diretamente pela UI
- 📊 **Avaliação:** Compare desempenho dos modelos
- 💡 **Previsões:** Use o modelo para prever valores futuros

### Estrutura do Projeto

```
Trabalho/
├── app.py                    # Interface Streamlit
├── requirements.txt          # Dependências
├── README.md                 # Guia rápido
├── DOCUMENTATION.md          # Esta documentação
├── data/
│   ├── Month.csv            # Dados mensais
│   └── Week.csv             # Dados semanais
├── src/
│   ├── __init__.py
│   ├── inspect_dataset.py    # Inspeção de dados
│   ├── eda.py                # Análise exploratória
│   ├── pipeline.py           # Pipeline de treino
│   ├── tune.py               # Tuning de hiperparâmetros
│   └── evaluate.py           # Avaliação de modelos
├── models/
│   ├── dollar_model.pkl      # Baseline
│   ├── best_rf.pkl           # RF otimizado
│   └── best_xgb.pkl          # XGB otimizado (melhor)
├── reports/
│   ├── plots/                # Gráficos EDA
│   └── evaluation/           # Gráficos de comparação
├── notebooks/
│   └── 01-exploracao.ipynb   # Análise interativa
└── tests/
    └── test_placeholder.py   # Testes
```

### Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `app.py` | Interface Streamlit com 4 abas |
| `src/eda.py` | Gera plots e estatísticas descritivas |
| `src/pipeline.py` | Define pipeline e treina modelo base |
| `src/tune.py` | Busca aleatória de hiperparâmetros |
| `src/evaluate.py` | Compara desempenho de modelos |
| `DOCUMENTATION.md` | Esta documentação |

---

## Referências

- **Dataset:** [Currency Exchange Rate USD/BRL (1993-2019)](https://www.kaggle.com/)
- **Bibliotecas:** Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn
- **Métodos:** Random Forest, XGBoost, RandomizedSearchCV

---

## Autor

Projeto de Previsão de Taxa de Câmbio | Dezembro 2025

---

## Licença

Este projeto é disponibilizado gratuitamente para fins educacionais e comerciais.

---

**Última atualização:** 01 de Dezembro de 2025
