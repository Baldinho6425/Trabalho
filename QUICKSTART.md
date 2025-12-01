# 🚀 INÍCIO RÁPIDO — Projeto USD/BRL

## ⚡ Primeira Vez? Comece Aqui!

### Passo 1: Instalar Dependências
```bash
cd c:/Users/eduar/Desktop/Trabalho
pip install -r requirements.txt
```

### Passo 2: Rodar Interface Streamlit (Recomendado)
```bash
streamlit run app.py
```

Abra no navegador: **http://localhost:8501**

---

## 🎯 O que Fazer no Streamlit

### Aba 1: 🔍 Exploração
1. Selecione "Month" ou "Week"
2. Clique "Executar EDA"
3. Veja gráficos de análise

### Aba 2: 🎯 Treino
1. Escolha modelo: Pipeline / RF Tune / XGB Tune
2. Defina iterações (10-30 recomendado)
3. Clique "Treinar"
4. Veja modelos disponíveis

### Aba 3: 📊 Avaliação
1. Clique "Executar Avaliação"
2. Compare MAE, RMSE, R² entre modelos
3. Veja gráficos de previsões

### Aba 4: 💡 Previsões
1. Insira valores: Opening, Max, Min
2. Selecione data (ano, mês, dia)
3. Clique "Prever"
4. Veja resultado com comparação histórica

---

## 📚 Documentação

| Arquivo | Para Quem |
|---------|-----------|
| **README.md** | Overview geral |
| **DOCUMENTATION.md** | Entender tudo em detalhes |
| **KAGGLE.md** | Publicar no Kaggle |
| **CHECKLIST.md** | Ver o que foi feito |

---

## 🧪 Testes Rápidos (sem Streamlit)

```bash
# Ver dados
python -m src.inspect_dataset.py --freq Month

# Análise exploratória
python -m src.eda --freq Month

# Treinar rapidinho
python -m src.pipeline --file data/Month.csv --target Last

# Treinar XGBoost (30 iterações)
python -m src.tune --file data/Month.csv --target Last --model xgb --n-iter 30

# Comparar modelos
python -m src.evaluate --file data/Month.csv --target Last
```

---

## ✨ Resultados Esperados

```
Baseline RF:   R² = 0.9957 ✅
Tuned RF:      R² = 0.9989 ✅✅
Tuned XGB:     R² = 1.0000 ⭐⭐⭐
```

---

## 📂 Arquivos Importantes

```
✅ app.py                    # Interface (USAR ISTO!)
✅ data/Month.csv           # Dados mensais
✅ models/best_xgb.pkl      # Melhor modelo
✅ reports/plots/           # Gráficos da análise
✅ reports/evaluation/      # Gráficos da comparação
```

---

## 🎓 Entender o Projeto

1. **Dados?** → Veja `DOCUMENTATION.md` (seção "Descrição do Dataset")
2. **Métodos?** → Veja `DOCUMENTATION.md` (seção "Metodologia")
3. **Resultados?** → Veja `DOCUMENTATION.md` (seção "Desempenho dos Modelos")
4. **Como funciona?** → Veja `README.md`

---

## 💡 Dicas

- 🎯 Use Streamlit para interagir (mais fácil)
- 📊 Veja os gráficos em `reports/` para análise
- 💾 Modelos já estão treinados em `models/`
- 📈 O melhor modelo é XGBoost (R² = 1.0)
- 🔄 Pode retreinar quando quiser via Streamlit

---

## 🆘 Problemas?

```bash
# Erro de import?
pip install --upgrade scikit-learn xgboost streamlit

# Erro de arquivo não encontrado?
cd c:/Users/eduar/Desktop/Trabalho
ls -la  # Verifique a estrutura

# Porta 8501 em uso?
streamlit run app.py --server.port 8502
```

---

## 📝 Próximos Passos

1. ✅ Explorar dados no Streamlit
2. ✅ Treinar novos modelos
3. ✅ Fazer previsões
4. 📖 Ler documentação (DOCUMENTATION.md)
5. 🚀 Publicar no Kaggle (siga KAGGLE.md)

---

**Pronto? Vá para http://localhost:8501!** 🚀

---

*Projeto concluído em 01 de Dezembro de 2025*
