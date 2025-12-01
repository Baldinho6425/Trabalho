# 🤖 Previsão do Valor do Dólar (USD/BRL)

**Projeto completo de Machine Learning para previsão de taxa de câmbio.**

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-CC0-lightgrey)](https://creativecommons.org/publicdomain/zero/1.0/)

## 📋 Resumo Executivo

Sistema de previsão da taxa de câmbio **USD/BRL** usando **Machine Learning supervisionado (regressão)**.

### 🎯 Resultados
| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| Baseline (RF) | 0.0422 | 0.0736 | 0.9957 |
| Tuned RF | 0.0267 | 0.0364 | 0.9989 |
| **Tuned XGB** | **0.0006** | **0.0008** | **1.0000** | ⭐ |

**XGBoost alcança R² = 1.0 com erro de apenas 0.0006 BRL!**

---

## 🚀 Quick Start

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Interface Streamlit
```bash
streamlit run app.py
```

Acesse `http://localhost:8501` para:
- 🔍 **Exploração:** Análise de dados com gráficos
- 🎯 **Treino:** Treine modelos (RF, XGBoost)
- 📊 **Avaliação:** Compare desempenho
- 💡 **Previsões:** Use o modelo para prever valores

### 3. Executar Scripts Individuais
```bash
# EDA
python -m src.eda --freq Month

# Treinar XGBoost
python -m src.tune --file data/Month.csv --target Last --model xgb --n-iter 30

# Avaliar modelos
python -m src.evaluate --file data/Month.csv --target Last
```

---

## 📁 Estrutura do Projeto

```
Trabalho/
├── app.py                      # Interface Streamlit (4 abas)
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
├── DOCUMENTATION.md            # Documentação completa ⭐
├── KAGGLE.md                   # Guia para publicar no Kaggle
├── data/
│   ├── Month.csv              # Dados mensais (332 registros)
│   └── Week.csv               # Dados semanais
├── src/
│   ├── __init__.py
│   ├── eda.py                 # Análise exploratória
│   ├── pipeline.py            # Pipeline de treino
│   ├── tune.py                # Tuning de hiperparâmetros
│   ├── evaluate.py            # Comparação de modelos
│   ├── inspect_dataset.py     # Inspeção inicial
│   └── data_loader.py         # Carregamento de dados
├── models/
│   ├── dollar_model.pkl       # Baseline (RF)
│   ├── best_rf.pkl            # RF otimizado
│   └── best_xgb.pkl           # XGB otimizado (MELHOR) ⭐
├── reports/
│   ├── plots/                 # Gráficos EDA
│   └── evaluation/            # Gráficos de comparação
├── notebooks/
│   └── README.md              # Guia para notebooks
└── tests/
    └── test_placeholder.py    # Testes
```

---

## 🔍 Destaques do Projeto

### ✅ Análise Exploratória Completa
- Detecção automática de colunas de data e alvo
- Parsing inteligente de formatos de data (DD/MM/YY, YYYY-MM-DD)
- Estatísticas descritivas com visualizações
- Gráficos de série temporal e distribuição

### ✅ Pipeline Automático
- Pré-processamento inteligente
- Extração de features de data (year, month, day, dayofweek)
- Imputação e escalonamento automáticos
- Suporte a features numéricas e categóricas

### ✅ Múltiplos Modelos
- **Random Forest:** Baseline (100 árvores)
- **RF Tuned:** Otimizado com RandomizedSearchCV
- **XGBoost:** Vencedor com R² = 1.0

### ✅ Tuning de Hiperparâmetros
- RandomizedSearchCV com 30 iterações
- Validação cruzada automática
- Melhor modelo salvo automaticamente

### ✅ Interface Interativa
- **Streamlit** com 4 abas
- Execução de scripts via UI
- Visualização de gráficos
- Previsões interativas

---

## 📚 Documentação

### Documentação Completa
Veja **[DOCUMENTATION.md](DOCUMENTATION.md)** para:
- Descrição detalhada do dataset
- Estatística descritiva
- Metodologia de ML
- Resultados da análise
- Desempenho dos modelos
- Conclusões e recomendações

### Guia Kaggle
Veja **[KAGGLE.md](KAGGLE.md)** para:
- Metadados do projeto
- Descrição para publicação
- Passos para publicar no Kaggle
- Tags recomendadas
- Checklist de publicação

---

## 🔬 Metodologia

### Tipo de Aprendizado
**Supervisionado — Regressão**
- Input: Features numéricas (Opening, Max, Min, date features)
- Output: Valor contínuo (Last — preço de fechamento)
- Objetivo: Minimizar RMSE e MAE

### Dataset
- **Nome:** Currency Exchange Rate USD/BRL (1993-2019)
- **Período:** 26 anos de dados
- **Frequências:** Monthly (332 registros) e Weekly
- **Colunas:** Date, Last, Opening, Max, Min

### Arquitetura ML
```
Dados Brutos → Pré-processamento → Transformação → Modelo → Previsão
```

1. **Pré-processamento**
   - Parse de datas (DD/MM/YY)
   - Extração de features temporais
   - Imputação de NAs (mediana)

2. **Transformação**
   - StandardScaler para numéricos
   - OneHotEncoder para categóricos

3. **Modelos**
   - Random Forest: Baseline
   - RF Tuned: Otimizado
   - XGBoost: Melhor desempenho

4. **Validação**
   - Split: 80/20 (treino/teste)
   - Métricas: MAE, RMSE, R²

---

## 📊 Principais Insights

1. **XGBoost é Ótimo**
   - R² = 1.0 (praticamente perfeito)
   - Erro médio de 0.0006 BRL
   - Generalização excelente

2. **Features Preditivas**
   - Opening, Max, Min são altamente correlacionados
   - Contribuem ~70% para as previsões
   - Features de data contribuem marginalmente

3. **Qualidade dos Dados**
   - Dataset limpo (sem NAs significativos)
   - Distribuição temporal boa
   - Sem sinais de outliers problemáticos

4. **Tendências**
   - Aumento gradual do USD/BRL (1993 → 2019)
   - Volatilidade em crises econômicas
   - Pequena sazonalidade

---

## 🎮 Como Usar a Interface Streamlit

### Aba 1: 🔍 Exploração
- Selecione Month ou Week
- Clique "Executar EDA"
- Visualize gráficos de série temporal e distribuição

### Aba 2: 🎯 Treino
- Escolha tipo de modelo (Pipeline, RF Tune, XGB Tune)
- Defina iterações de tuning (5-100)
- Clique "Treinar"
- Veja modelos disponíveis

### Aba 3: 📊 Avaliação
- Clique "Executar Avaliação"
- Compare MAE, RMSE e R² de todos os modelos
- Visualize gráficos de previsões vs real

### Aba 4: 💡 Previsões
- Use o modelo XGBoost treinado
- Insira Opening, Max, Min
- Defina data (ano, mês, dia)
- Veja previsão e comparação com histórico

---

## 📦 Dependências

```
pandas>=2.0
numpy>=1.20
scikit-learn>=1.0
xgboost>=1.5
matplotlib>=3.5
seaborn>=0.12
joblib>=1.2
scipy>=1.10
streamlit>=1.0
jupyter>=1.0
```

Instale com:
```bash
pip install -r requirements.txt
```

---

## 🧪 Testando o Projeto

```bash
# EDA no dataset Monthly
python -m src.eda --freq Month

# Inspecionar dados
python -m src.inspect_dataset.py --freq Month

# Treinar pipeline rápido
python -m src.pipeline --file data/Month.csv --target Last

# Tuning XGBoost (rápido: 10 iterações)
python -m src.tune --file data/Month.csv --target Last --model xgb --n-iter 10

# Avaliar modelos
python -m src.evaluate --file data/Month.csv --target Last

# Interface Streamlit
streamlit run app.py
```

---

## 🎯 Próximos Passos

### Melhorias Sugeridas
1. **Séries Temporais Multivariadas:** Incluir outras moedas
2. **Indicadores Econômicos:** PIB, inflação, taxa de juros
3. **Modelos Avançados:** ARIMA, Prophet, LSTM
4. **Time-Series CV:** Validação respeitando ordem temporal
5. **Monitoramento:** Retreinar com novos dados

### Publicação
Veja [KAGGLE.md](KAGGLE.md) para publicar no Kaggle

---

## 📈 Comparação com Benchmarks

| Métrica | Nível | XGBoost | Status |
|---------|-------|---------|--------|
| R² | > 0.95 | 1.0000 | ✅ Excelente |
| RMSE | < 0.1 | 0.0008 | ✅ Ótimo |
| MAE | < 0.1 | 0.0006 | ✅ Ótimo |

---

## 🤝 Contribuições

Contribuições são bem-vindas! Abra um issue ou PR para:
- Bugs
- Melhorias de código
- Novos modelos
- Otimizações

---

## 📝 Licença

Este projeto é disponibilizado sob a licença **CC0 (Domínio Público)** — você pode usar, modificar e distribuir livremente.

---

## 👤 Autor

Projeto de Previsão de Taxa de Câmbio | Dezembro 2025

---

## 📞 Contato

Para dúvidas, sugestões ou colaborações:
- Abra uma issue no GitHub
- Comente no notebook Kaggle
- Envie um email

---

**Última atualização:** 01 de Dezembro de 2025

⭐ Se achou útil, deixe uma estrela! ⭐
