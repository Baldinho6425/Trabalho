"""Exploração de dados (EDA) para o dataset USD/BRL.

Gera um resumo estatístico e alguns gráficos básicos, e salva plots em `reports/plots/`.
Uso:
  python src/eda.py --file data/Month.csv
  python src/eda.py --freq Month
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from src.inspect_dataset import find_file_by_freq


def ensure_reports_dir():
    d = Path(__file__).resolve().parents[1] / 'reports' / 'plots'
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    return pd.read_csv(path)


def basic_stats(df: pd.DataFrame) -> str:
    buf = []
    buf.append(f"Shape: {df.shape}")
    buf.append('\nColumns:')
    buf.append(', '.join(df.columns.tolist()))
    buf.append('\n\nDtypes:')
    buf.append(str(df.dtypes))
    buf.append('\n\nDescribe:')
    buf.append(str(df.describe(include='all')))
    return '\n'.join(buf)


def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str, out_dir: Path):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x=date_col, y=value_col)
    plt.title(f'{value_col} over time')
    p = out_dir / f'{value_col}_timeseries.png'
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def plot_hist(df: pd.DataFrame, col: str, out_dir: Path):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Histograma: {col}')
    p = out_dir / f'{col}_hist.png'
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def find_date_and_value(df: pd.DataFrame):
    date_candidates = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower() or 'time' in c.lower()]
    # incluir 'last' e variações nas candidatas de valor
    value_keywords = ['usd', 'dollar', 'brl', 'rate', 'exchange', 'close', 'price', 'valor', 'last']
    value_candidates = [c for c in df.columns if any(k in c.lower() for k in value_keywords)]
    return date_candidates, value_candidates


def main():
    parser = argparse.ArgumentParser(description='EDA para dataset USD/BRL')
    parser.add_argument('--freq', choices=['Month', 'Week'], help='Escolhe Month ou Week em data/')
    parser.add_argument('--file', help='Caminho direto para o arquivo')
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / 'data'
    if args.file:
        p = Path(args.file)
    elif args.freq:
        p = find_file_by_freq(args.freq, data_dir)
        if not p:
            print('Arquivo não encontrado', file=sys.stderr)
            sys.exit(1)
    else:
        print('Forneça --file ou --freq', file=sys.stderr)
        sys.exit(1)

    df = load(p)
    out_dir = ensure_reports_dir()

    print('\n--- Estatísticas básicas ---\n')
    print(basic_stats(df))

    date_cands, value_cands = find_date_and_value(df)
    print('\nSugestões de coluna de data:', date_cands)
    print('Sugestões de coluna alvo:', value_cands)

    # Tentar converter primeira data candidata usando formatos comuns
    date_col = date_cands[0] if date_cands else None
    value_col = value_cands[0] if value_cands else None

    if date_col:
        parsed = False
        for fmt in ('%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d'):
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                if df[date_col].notna().sum() / max(1, len(df)) > 0.5:
                    parsed = True
                    break
            except Exception:
                continue
        if not parsed:
            # fallback para parsing flexível
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Se não detectou coluna alvo automática, escolher primeira coluna numérica (exceto data)
    if not value_col:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if date_col and date_col in num_cols:
            num_cols.remove(date_col)
        if num_cols:
            value_col = num_cols[0]

    if date_col and value_col:
        print(f'Gerando plots para `{date_col}` x `{value_col}` em {out_dir}')
        # ordenar por data
        df2 = df.sort_values(by=date_col).dropna(subset=[date_col, value_col])
        plot_time_series(df2, date_col, value_col, out_dir)
        plot_hist(df2, value_col, out_dir)
    else:
        print('Não há colunas detectadas suficientes para gerar plots de série temporal.')

    print(f'Plots salvos em: {out_dir}')


if __name__ == '__main__':
    main()
