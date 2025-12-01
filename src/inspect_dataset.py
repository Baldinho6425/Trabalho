"""Inspeção rápida para os arquivos do dataset 'Month' e 'Week'.

Uso:
  python src/inspect_dataset.py --freq Month
  python src/inspect_dataset.py --file data/Month.csv

O script busca por arquivos em `data/` cujo nome contenha 'month' ou 'week' (case-insensitive),
lê o CSV e exibe shape, colunas, as primeiras linhas e sugestões de colunas de data/target.
"""
from pathlib import Path
import argparse
import pandas as pd
import sys


def find_file_by_freq(freq: str, data_dir: Path) -> Path | None:
    freq = freq.lower()
    for ext in ('.csv', '.xls', '.xlsx'):
        # procurar arquivos contendo a palavra freq
        candidates = list(data_dir.glob(f'*{freq}*{ext}'))
        if candidates:
            return candidates[0]
    # tentar sem extensão
    candidates = list(data_dir.glob(f'*{freq}*'))
    return candidates[0] if candidates else None


def suggest_date_columns(df: pd.DataFrame) -> list:
    candidates = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()]
    # detectar colunas convertíveis para datetime
    for c in df.columns:
        if c in candidates:
            continue
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            non_na = parsed.notna().sum()
            if non_na / max(1, len(parsed)) > 0.6:
                candidates.append(c)
        except Exception:
            pass
    return candidates


def suggest_target_columns(df: pd.DataFrame) -> list:
    keywords = ['usd', 'dollar', 'brl', 'rate', 'exchange', 'close', 'price', 'valor']
    cols = [c for c in df.columns if any(k in c.lower() for k in keywords)]
    return cols


def inspect_path(path: Path) -> None:
    print(f'Lendo: {path}')
    try:
        if path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f'Erro ao ler arquivo: {e}', file=sys.stderr)
        return

    print('Shape:', df.shape)
    print('Colunas:', df.columns.tolist())
    print('\nPrimeiras linhas:')
    print(df.head().to_string())
    date_cands = suggest_date_columns(df)
    target_cands = suggest_target_columns(df)
    print('\nSugestões de coluna(s) de data:', date_cands)
    print('Sugestões de coluna(s) alvo:', target_cands)


def main():
    parser = argparse.ArgumentParser(description='Inspecionar arquivos Month/Week do dataset USD/BRL')
    parser.add_argument('--freq', choices=['Month', 'Week'], help='Escolhe Month ou Week e procura em data/')
    parser.add_argument('--file', help='Caminho direto para o arquivo de dados')
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / 'data'
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print('Arquivo não encontrado:', p, file=sys.stderr)
            sys.exit(1)
        inspect_path(p)
        return

    if args.freq:
        p = find_file_by_freq(args.freq, data_dir)
        if not p:
            print(f'Nenhum arquivo encontrado para freq {args.freq} em {data_dir}', file=sys.stderr)
            sys.exit(1)
        inspect_path(p)
        return

    print('Forneça `--freq Month|Week` ou `--file <caminho>`')


if __name__ == '__main__':
    main()
