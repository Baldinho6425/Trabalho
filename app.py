import streamlit as st
import subprocess
import joblib
import pandas as pd
import numpy as np
import calendar
from pathlib import Path

st.set_page_config(page_title='USD/BRL Forecast', layout='wide')

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'

st.title('🤖 Previsão USD/BRL — Painel Completo')

st.markdown('''
Sistema de previsão do valor do Dólar (USD) em relação ao Real (BRL) usando Machine Learning.
Inclui EDA, treinamento de modelos e ajuste de hiperparâmetros.
''')

# Abas principais
tab1, tab2, tab3, tab4 = st.tabs(['🔍 Exploração', '🎯 Treino', '📊 Avaliação', '💡 Previsões'])

# Helper functions
def run_cmd(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, cwd=str(ROOT))
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return 1, '', str(e)

# ===== TAB 1: EXPLORAÇÃO =====
with tab1:
    st.header('Exploração de Dados (EDA)')
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        freq = st.selectbox('Frequência dos dados', ['Month', 'Week'])
        if st.button('Executar EDA'):
            st.info('Rodando análise exploratória...')
            rc, out, err = run_cmd(['python', '-m', 'src.eda', '--freq', freq])
            if rc == 0:
                st.success('EDA concluída!')
                if out:
                    st.code(out, language='text')
            else:
                st.error(f'Erro: {err}')
    
    with col2:
        # Mostrar plots gerados
        rpt = REPORTS_DIR / 'plots'
        if rpt.exists():
            imgs = list(rpt.glob('*.png'))
            if imgs:
                st.subheader('Gráficos Gerados')
                cols = st.columns(2)
                for i, img in enumerate(imgs[:6]):
                    with cols[i % 2]:
                        st.image(str(img), caption=img.name, use_column_width=True)
            else:
                st.info('Nenhum gráfico gerado ainda. Clique em "Executar EDA".')
        else:
            st.info('Pasta reports/plots não existe. Execute EDA primeiro.')

# ===== TAB 2: TREINO =====
with tab2:
    st.header('Treinamento de Modelos')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader('Configuração')
        model_type = st.radio('Tipo de modelo', ['Pipeline Rápido', 'Tuning RF', 'Tuning XGB'])
        freq = st.selectbox('Frequência', ['Month', 'Week'], key='train_freq')
        target = st.text_input('Coluna alvo (deixe vazio para auto-detectar)', value='Last')
        n_iter = st.number_input('Iterações tuning', min_value=5, max_value=100, value=20)
        
        if st.button('▶ Treinar'):
            st.info('Treinando modelo...')
            
            # Encontrar arquivo
            data_file = None
            for ext in ('.csv', '.xls', '.xlsx'):
                candidates = list(DATA_DIR.glob(f'*{freq}*{ext}'))
                if candidates:
                    data_file = str(candidates[0])
                    break
            
            if not data_file:
                st.error(f'Arquivo {freq} não encontrado em data/')
            else:
                if model_type == 'Pipeline Rápido':
                    cmd = ['python', '-m', 'src.pipeline', '--file', data_file, '--target', target, '--out', 'models/dollar_model.pkl']
                elif model_type == 'Tuning RF':
                    cmd = ['python', '-m', 'src.tune', '--file', data_file, '--target', target, '--model', 'rf', '--n-iter', str(n_iter), '--out', 'models/best_rf.pkl']
                else:  # Tuning XGB
                    cmd = ['python', '-m', 'src.tune', '--file', data_file, '--target', target, '--model', 'xgb', '--n-iter', str(n_iter), '--out', 'models/best_xgb.pkl']
                
                rc, out, err = run_cmd(cmd)
                if rc == 0:
                    st.success('Treinamento concluído!')
                    st.code(out, language='text')
                else:
                    st.error(f'Erro: {err}')
    
    with col2:
        st.subheader('Modelos Disponíveis')
        if MODELS_DIR.exists():
            models = list(MODELS_DIR.glob('*.pkl'))
            if models:
                for m in models:
                    st.info(f'✅ {m.name}')
            else:
                st.warning('Nenhum modelo treinado ainda.')
        else:
            st.warning('Pasta models não existe.')

# ===== TAB 3: AVALIAÇÃO =====
with tab3:
    st.header('Comparação de Modelos')
    
    if st.button('📈 Executar Avaliação'):
        st.info('Avaliando modelos...')
        
        rc, out, err = run_cmd(['python', '-m', 'src.evaluate', '--file', 'data/Month.csv', '--target', 'Last', '--out', 'reports/evaluation'])
        
        if rc == 0:
            st.success('Avaliação concluída!')
            st.code(out, language='text')
            
            # Mostrar plots de avaliação
            eval_dir = REPORTS_DIR / 'evaluation'
            if eval_dir.exists():
                imgs = list(eval_dir.glob('*.png'))
                if imgs:
                    st.subheader('Gráficos de Comparação')
                    for img in imgs:
                        st.image(str(img), caption=img.name, use_column_width=True)
        else:
            st.error(f'Erro: {err}')
    
    st.markdown('---')
    st.info('''
    **Resultados esperados (Month.csv):**
    - Baseline (RF): R² ≈ 0.996
    - Tuned RF: R² ≈ 0.999
    - Tuned XGB: R² ≈ 1.000 ⭐
    ''')

# ===== TAB 4: PREVISÕES =====
with tab4:
    st.header('💡 Fazer Previsões')
    
    # Carregar modelo
    model_path = MODELS_DIR / 'best_xgb.pkl'
    
    if not model_path.exists():
        st.warning('⚠️ Modelo XGBoost não encontrado. Treie primeiro usando a aba "Treino".')
    else:
        st.success(f'✅ Modelo carregado: {model_path.name}')
        
        # Carregar dados históricos
        data_file = DATA_DIR / 'Month.csv'
        if data_file.exists():
            df = pd.read_csv(data_file)
            st.subheader('Histórico de Dados')
            st.dataframe(df.head(10), use_container_width=True)
            
            st.info(f'📊 Total de registros: {len(df)}')
            
            # Carregar e usar modelo
            try:
                model = joblib.load(model_path)
                
                st.subheader('Entrada para Previsão')
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    opening = st.number_input('Abertura (Opening)', value=2.5, min_value=0.0, step=0.01)
                with col2:
                    max_val = st.number_input('Máximo (Max)', value=2.6, min_value=0.0, step=0.01)
                with col3:
                    min_val = st.number_input('Mínimo (Min)', value=2.4, min_value=0.0, step=0.01)
                with col4:
                    year = st.number_input('Ano', value=2024, min_value=1990, max_value=2100)
                
                month = st.slider('Mês', min_value=1, max_value=12, value=12)
                day = st.slider('Dia', min_value=1, max_value=31, value=15)
                
                if st.button('🔮 Prever'):
                    # Preparar dados
                    input_data = pd.DataFrame({
                        'Opening': [opening],
                        'Max': [max_val],
                        'Min': [min_val],
                        'year': [year],
                        'month': [month],
                        'day': [day],
                        'dayofweek': [pd.Timestamp(year, month, day).dayofweek]
                    })
                    
                    prediction = model.predict(input_data)[0]
                    
                    st.success(f'### 💰 Previsão: USD/BRL = **{prediction:.4f}**')
                    
                    # Comparação com histórico
                    hist_mean = df['Last'].mean()
                    hist_std = df['Last'].std()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Média Histórica', f'{hist_mean:.4f}')
                    with col2:
                        st.metric('Desvio Padrão', f'{hist_std:.4f}')
                    with col3:
                        diff = prediction - hist_mean
                        st.metric('Diferença da Média', f'{diff:.4f}')

                st.markdown('---')
                st.subheader('Previsão Anual (média mensal)')
                annual_year = st.number_input('Ano para previsão anual', value=2025, min_value=1990, max_value=2100, key='annual_year')

                if st.button('🔮 Prever Ano'):
                    try:
                        # Tentar parsear datas históricas para agrupar por mês
                        df_copy = df.copy()
                        if 'Date' in df_copy.columns:
                            df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')
                        else:
                            df_copy['Date'] = pd.NaT

                        # Calcular mediana histórica por mês para Opening/Max/Min
                        month_stats = df_copy.groupby(df_copy['Date'].dt.month)[['Opening', 'Max', 'Min']].median()

                        rows = []
                        for m in range(1, 13):
                            if m in month_stats.index:
                                opening_m = month_stats.loc[m, 'Opening']
                                max_m = month_stats.loc[m, 'Max']
                                min_m = month_stats.loc[m, 'Min']
                            else:
                                # fallback para mediana global
                                opening_m = df['Opening'].median()
                                max_m = df['Max'].median()
                                min_m = df['Min'].median()

                            day_m = 15
                            dow = pd.Timestamp(int(annual_year), int(m), int(day_m)).dayofweek
                            rows.append({
                                'Opening': opening_m,
                                'Max': max_m,
                                'Min': min_m,
                                'year': int(annual_year),
                                'month': int(m),
                                'day': int(day_m),
                                'dayofweek': int(dow)
                            })

                        input_annual = pd.DataFrame(rows)
                        preds = model.predict(input_annual)

                        # Preparar tabela de resultados
                        months = [calendar.month_name[m] for m in range(1, 13)]
                        results_df = pd.DataFrame({'Month': months, 'Predicted_Last': preds})

                        annual_mean = float(np.mean(preds))

                        st.success(f'### 📅 Previsão média anual para {annual_year}: **{annual_mean:.4f}**')

                        st.subheader('Previsões Mensais')
                        st.table(results_df)
                        st.line_chart(results_df.set_index('Month')['Predicted_Last'])

                        # Comparação com histórico
                        hist_mean = df['Last'].mean()
                        diff = annual_mean - hist_mean
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric('Média Histórica (Last)', f'{hist_mean:.4f}')
                        with col2:
                            st.metric(f'Diferença (Annual - Hist)', f'{diff:.4f}')

                    except Exception as e:
                        st.error(f'Erro ao gerar previsão anual: {e}')
            
            except Exception as e:
                st.error(f'Erro ao carregar modelo: {e}')
        else:
            st.error('Arquivo data/Month.csv não encontrado.')

st.markdown('---')
st.markdown('''
**Como usar:**
1. 🔍 **Exploração**: Execute EDA para analisar os dados
2. 🎯 **Treino**: Treine modelos usando diferentes estratégias
3. 📊 **Avaliação**: Compare o desempenho dos modelos
4. 💡 **Previsões**: Use o melhor modelo para fazer previsões

**Dataset:** Currency Exchange Rate USD/BRL (1993-2019)
''')
