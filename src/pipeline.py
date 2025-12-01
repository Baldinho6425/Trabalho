"""Pipeline de pré-processamento e treino para previsão do USD/BRL.

Fornece funções para construir pipeline com imputação, features de data, codificação e escalonamento,
e treinar um modelo de regressão supervisionado (RandomForest por padrão).
"""
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def detect_date_and_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    date_candidates = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()]
    target_candidates = [c for c in df.columns if any(k in c.lower() for k in ['usd', 'dollar', 'brl', 'rate', 'exchange', 'close', 'price', 'valor'])]
    date_col = date_candidates[0] if date_candidates else None
    target_col = target_candidates[0] if target_candidates else None
    return date_col, target_col


def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df = df.drop(columns=[date_col])
    return df


def build_feature_sets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    # separar colunas numéricas e categóricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return X, y, num_cols, cat_cols


def get_pipeline(num_cols: List[str], cat_cols: List[str], model=None) -> Pipeline:
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ])

    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipe = Pipeline([('preproc', preprocessor), ('model', model)])
    return pipe


def train_and_evaluate(df: pd.DataFrame, target_col: str, date_col: Optional[str] = None, model_out: str = 'models/dollar_model.pkl') -> dict:
    if date_col:
        df = add_date_features(df, date_col)

    X, y, num_cols, cat_cols = build_feature_sets(df, target_col)
    pipe = get_pipeline(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Compute RMSE in a way compatible with different sklearn versions
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    metrics = {
        'mae': mean_absolute_error(y_test, preds),
        'rmse': rmse,
        'r2': r2_score(y_test, preds)
    }

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Treinar pipeline básico para USD/BRL')
    parser.add_argument('--file', required=True, help='Caminho para CSV')
    parser.add_argument('--date-col', help='Nome da coluna de data (opcional)')
    parser.add_argument('--target', help='Nome da coluna alvo (opcional)')
    parser.add_argument('--out', default='models/dollar_model.pkl')
    args = parser.parse_args()

    df = pd.read_csv(args.file) if Path(args.file).suffix.lower() != '.xlsx' else pd.read_excel(args.file)

    date_col = args.date_col
    target_col = args.target
    if not date_col or not target_col:
        dcol, tcol = detect_date_and_target(df)
        date_col = date_col or dcol
        target_col = target_col or tcol

    if not target_col:
        raise ValueError('Nenhuma coluna alvo detectada; especifique via --target')

    metrics = train_and_evaluate(df, target_col, date_col, model_out=args.out)
    print('Métricas:', metrics)
