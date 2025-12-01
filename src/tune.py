"""Ajuste de hiperparâmetros para modelos de regressão (RandomForest e XGBoost).

Uso:
  python src/tune.py --file data/Month.csv --model rf
  python src/tune.py --file data/Month.csv --model xgb
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import randint as sp_randint, uniform as sp_uniform
import joblib

from src.pipeline import detect_date_and_target, add_date_features, build_feature_sets, get_pipeline


def tune_rf(pipe, X, y, n_iter=20, cv=3, random_state=42):
    param_dist = {
        'model__n_estimators': sp_randint(50, 500),
        'model__max_depth': sp_randint(3, 20),
        'model__min_samples_split': sp_randint(2, 20),
        'model__min_samples_leaf': sp_randint(1, 20),
    }
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='neg_root_mean_squared_error', random_state=random_state, n_jobs=-1)
    search.fit(X, y)
    return search


def tune_xgb(pipe, X, y, n_iter=20, cv=3, random_state=42):
    param_dist = {
        'model__n_estimators': sp_randint(50, 500),
        'model__max_depth': sp_randint(3, 12),
        'model__learning_rate': sp_uniform(0.01, 0.3),
        'model__subsample': sp_uniform(0.5, 0.5),
    }
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='neg_root_mean_squared_error', random_state=random_state, n_jobs=-1)
    search.fit(X, y)
    return search


def main():
    parser = argparse.ArgumentParser(description='Ajuste de hiperparâmetros para modelos de regressão')
    parser.add_argument('--file', required=True, help='Caminho para CSV')
    parser.add_argument('--date-col', help='Coluna de data (opcional)')
    parser.add_argument('--target', help='Coluna alvo (opcional)')
    parser.add_argument('--model', choices=['rf', 'xgb'], default='rf')
    parser.add_argument('--out', default='models/best_model.pkl')
    parser.add_argument('--n-iter', type=int, default=20)
    args = parser.parse_args()

    path = Path(args.file)
    if path.suffix.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    date_col = args.date_col
    target_col = args.target
    if not date_col or not target_col:
        dcol, tcol = detect_date_and_target(df)
        date_col = date_col or dcol
        target_col = target_col or tcol

    if date_col:
        df = add_date_features(df, date_col)

    if not target_col:
        raise ValueError('Não foi possível detectar a coluna alvo; informe via --target')

    X, y, num_cols, cat_cols = build_feature_sets(df, target_col)

    if args.model == 'rf':
        model = RandomForestRegressor(random_state=42)
        pipe = get_pipeline(num_cols, cat_cols, model=model)
        search = tune_rf(pipe, X, y, n_iter=args.n_iter)
    else:
        model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
        pipe = get_pipeline(num_cols, cat_cols, model=model)
        search = tune_xgb(pipe, X, y, n_iter=args.n_iter)

    print('Melhor score (neg RMSE):', search.best_score_)
    print('Melhores parâmetros:', search.best_params_)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out)
    print(f'Melhor modelo salvo em: {out}')


if __name__ == '__main__':
    main()
