"""Template de pré-processamento para o dataset do Dólar."""
import pandas as pd
from typing import Tuple


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Exemplo genérico: procurar coluna alvo 'close' ou 'price'
    target_candidates = [c for c in ['close', 'price', 'valor'] if c in df.columns]
    if not target_candidates:
        raise ValueError("Nenhuma coluna alvo encontrada. Procure por 'close', 'price' ou 'valor'.")
    target_col = target_candidates[0]

    # Exemplo de limpeza mínima
    df = df.copy()
    # Converter datas se houver coluna 'date' ou similar
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')

    # Simples preenchimento de NA
    df = df.fillna(method='ffill').fillna(method='bfill')

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


if __name__ == '__main__':
    print('Este módulo fornece `preprocess_df(df)` para ser usado pelo pipeline.')
