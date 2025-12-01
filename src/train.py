"""Template de treinamento — exemplo com RandomForestRegressor."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.data_loader import load_csv
from src.preprocess import preprocess_df


def train(path: str, model_out: str = 'models/dollar_model.pkl') -> None:
    df = load_csv(path)
    X, y = preprocess_df(df)

    # Simples split e treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.fillna(0), y_train)

    preds = model.predict(X_test.fillna(0))
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f'RMSE: {rmse:.4f}')

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f'Modelo salvo em: {out}')


def main():
    parser = argparse.ArgumentParser(description='Treinar modelo simples para prever dólar')
    parser.add_argument('--path', required=True, help='Caminho para CSV de entrada')
    parser.add_argument('--out', default='models/dollar_model.pkl', help='Caminho para salvar o modelo')
    args = parser.parse_args()
    train(args.path, args.out)


if __name__ == '__main__':
    main()
