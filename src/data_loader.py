"""Carregador simples de CSV para o projeto de previsão do Dólar."""
from pathlib import Path
import argparse
import pandas as pd


def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def main():
    parser = argparse.ArgumentParser(description="Carrega um CSV e mostra informações iniciais.")
    parser.add_argument("--path", required=True, help="Caminho para o arquivo CSV")
    parser.add_argument("--date-cols", nargs="*", help="Colunas a serem parseadas como datas")
    args = parser.parse_args()

    df = load_csv(args.path, parse_dates=args.date_cols)
    print("Shape:", df.shape)
    print("Colunas:", df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    main()
