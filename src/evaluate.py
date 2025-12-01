"""Avaliação e comparação de modelos treinados.

Uso:
  python -m src.evaluate --file data/Month.csv --target Last
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.pipeline import detect_date_and_target, add_date_features, build_feature_sets


def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = (mean_squared_error(y_test, preds)) ** 0.5
    r2 = r2_score(y_test, preds)
    return {'name': name, 'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': preds}


def plot_comparison(results, y_test, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Métricas
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [r['name'] for r in results]
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    r2s = [r['r2'] for r in results]

    axes[0].bar(names, maes)
    axes[0].set_title('MAE')
    axes[0].set_ylabel('Erro Absoluto Médio')

    axes[1].bar(names, rmses, color='orange')
    axes[1].set_title('RMSE')
    axes[1].set_ylabel('Raiz do Erro Quadrado Médio')

    axes[2].bar(names, r2s, color='green')
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('Coeficiente de Determinação')
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_comparison.png')
    plt.close()

    # Plot 2: Previsões vs real
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for i, r in enumerate(results):
        preds = r['preds']
        axes[i].scatter(range(len(y_test)), y_test, label='Real', alpha=0.6)
        axes[i].plot(range(len(y_test)), preds, label='Previsão', color='red', linewidth=2)
        axes[i].set_title(f'{r["name"]} - Previsões vs Real')
        axes[i].set_ylabel('Valor do Dólar')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'predictions_comparison.png')
    plt.close()

    print(f'Plots salvos em: {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Avaliar e comparar modelos')
    parser.add_argument('--file', required=True, help='Caminho para CSV')
    parser.add_argument('--target', required=True, help='Coluna alvo')
    parser.add_argument('--date-col', help='Coluna de data (opcional)')
    parser.add_argument('--out', default='reports/evaluation')
    args = parser.parse_args()

    path = Path(args.file)
    if path.suffix.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    date_col = args.date_col
    if not date_col:
        dcol, _ = detect_date_and_target(df)
        date_col = dcol

    if date_col:
        df = add_date_features(df, date_col)

    X, y, _, _ = build_feature_sets(df, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tentar carregar modelos
    models_dir = Path('models')
    results = []

    for model_path in [models_dir / 'dollar_model.pkl', models_dir / 'best_rf.pkl', models_dir / 'best_xgb.pkl']:
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                name = model_path.stem.replace('dollar_model', 'Baseline (RF)').replace('best_rf', 'Tuned RF').replace('best_xgb', 'Tuned XGB')
                res = evaluate_model(model, X_test, y_test, name)
                results.append(res)
                print(f'{name}: MAE={res["mae"]:.6f}, RMSE={res["rmse"]:.6f}, R²={res["r2"]:.6f}')
            except Exception as e:
                print(f'Erro ao carregar {model_path}: {e}')

    if results:
        plot_comparison(results, y_test, args.out)
        print(f'Avaliação concluída. Plots em: {args.out}')
    else:
        print('Nenhum modelo encontrado em models/')


if __name__ == '__main__':
    main()
