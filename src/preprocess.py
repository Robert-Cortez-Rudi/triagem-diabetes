"""Pré-processamento do dataset de triagem de diabetes (BRFSS 2015).

Etapas:
    1. Carga do CSV bruto.
    2. Remoção de registros duplicados.
    3. Descarte das variáveis não utilizadas.
    4. Feature engineering: ``InNaturaFood`` = Fruits OU Veggies.
    5. Divisão treino/teste estratificada (80/20, ``random_state=42``).

Uso pelo terminal::

    python -m src.preprocess
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"

TARGET: str = "Diabetes_012"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

CLASS_LABELS: dict[int, str] = {
    0: "Sem diabetes",
    1: "Pré-diabetes",
    2: "Diabetes",
}

# O CSV original grafa "AnyHealthcare"; as outras grafias cobrem variações do dataset.
DROP_COLUMNS: tuple[str, ...] = (
    "MentHlth",
    "Sex",
    "CholCheck",
    "AnyHealthcare",
    "NoDocbcCost",
)
DROP_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "AnyHealthcare": ("AnyHealthcare", "AnyHealthCare", "AnyHealtcare"),
}

FOOD_SOURCE_COLUMNS: tuple[str, str] = ("Fruits", "Veggies")
NEW_FEATURE: str = "InNaturaFood"


@dataclass(frozen=True)
class PreprocessResult:
    """Saída do pipeline de pré-processamento."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        """Lista ordenada das features usadas pelo modelo."""
        return list(self.X_train.columns)


# --------------------------------------------------------------------------- #
# Etapas
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV bruto do BRFSS 2015.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os dados originais.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo estiver vazio ou sem a coluna target.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"O arquivo está vazio: {path}") from exc

    if TARGET not in df.columns:
        raise ValueError(f"Coluna target '{TARGET}' ausente no dataset.")

    logger.info("Dataset carregado de %s", path.name)
    logger.info("Shape original: %s linhas x %s colunas", *df.shape)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas, considerando todas as colunas originais.

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame sem duplicatas e com índice reiniciado.
    """
    n_dup = int(df.duplicated().sum())
    pct = n_dup / len(df) if len(df) else 0.0
    logger.info("Registros duplicados encontrados: %d (%.2f%%)", n_dup, pct * 100)
    return df.drop_duplicates().reset_index(drop=True)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as 5 variáveis descartadas pelas regras de negócio.

    Aceita as grafias alternativas de ``AnyHealthcare``. Uma coluna que não
    aparece no DataFrame gera aviso, não erro.

    Args:
        df: DataFrame com as colunas originais.

    Returns:
        DataFrame sem as colunas descartadas.
    """
    to_drop: list[str] = []
    for col in DROP_COLUMNS:
        candidates = DROP_COLUMN_ALIASES.get(col, (col,))
        found = [c for c in candidates if c in df.columns]
        if found:
            to_drop.extend(found)
        else:
            logger.warning("Coluna a descartar não encontrada: %s", col)

    logger.info("Colunas descartadas (%d): %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop)


def add_in_natura_food(df: pd.DataFrame) -> pd.DataFrame:
    """Cria ``InNaturaFood`` a partir de ``Fruits`` e ``Veggies`` e remove as originais.

    ``InNaturaFood`` vale 1 se a pessoa consome frutas OU vegetais diariamente,
    e 0 se não consome nenhum dos dois. Serve tanto para o treino quanto para
    a entrada do usuário no ``app.py``.

    Args:
        df: DataFrame contendo ``Fruits`` e ``Veggies``.

    Returns:
        DataFrame com ``InNaturaFood`` e sem ``Fruits``/``Veggies``.

    Raises:
        KeyError: Se ``Fruits`` ou ``Veggies`` estiverem ausentes.
    """
    missing = [c for c in FOOD_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas necessárias para {NEW_FEATURE} ausentes: {missing}")

    out = df.copy()
    fruits, veggies = FOOD_SOURCE_COLUMNS
    out[NEW_FEATURE] = ((out[fruits] == 1) | (out[veggies] == 1)).astype(int)
    out = out.drop(columns=list(FOOD_SOURCE_COLUMNS))
    logger.info(
        "Feature criada: %s (%.2f%% = 1); removidas: %s",
        NEW_FEATURE,
        out[NEW_FEATURE].mean() * 100,
        list(FOOD_SOURCE_COLUMNS),
    )
    return out


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza, descarte de colunas e feature engineering.

    Ordem: duplicatas -> descarte de colunas -> ``InNaturaFood`` -> target como int.
    As duplicatas são removidas sobre as colunas originais, antes do descarte.

    Args:
        df: DataFrame bruto vindo de :func:`load_raw_data`.

    Returns:
        DataFrame limpo, pronto para a divisão treino/teste.
    """
    out = remove_duplicates(df)
    out = drop_unused_columns(out)
    out = add_in_natura_food(out)
    out[TARGET] = out[TARGET].astype(int)

    logger.info("Shape após limpeza: %s linhas x %s colunas", *out.shape)
    return out


def log_class_distribution(y: pd.Series, title: str) -> pd.DataFrame:
    """Registra no log a contagem e a proporção de cada classe do target.

    Args:
        y: Série com o target.
        title: Rótulo do conjunto (ex.: "Dataset limpo", "Treino").

    Returns:
        DataFrame com as colunas ``count`` e ``proportion`` por classe.
    """
    counts = y.value_counts().sort_index()
    dist = pd.DataFrame({"count": counts, "proportion": counts / counts.sum()})

    logger.info("Distribuição do target - %s (n=%d):", title, len(y))
    for cls, row in dist.iterrows():
        marker = "  <-- classe minoritária (atenção)" if cls == 1 else ""
        logger.info(
            "  %d - %-13s %8d  (%6.2f%%)%s",
            cls,
            CLASS_LABELS.get(int(cls), "?"),
            row["count"],
            row["proportion"] * 100,
            marker,
        )
    return dist


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessResult:
    """Divide em treino e teste com amostragem estratificada pelo target.

    Args:
        df: DataFrame limpo, contendo o target.
        test_size: Fração destinada ao teste (padrão 0.20).
        random_state: Semente de aleatoriedade (padrão 42).

    Returns:
        :class:`PreprocessResult` com X/y de treino e teste.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Divisão estratificada: treino=%d (%.0f%%), teste=%d (%.0f%%)",
        len(X_train),
        (1 - test_size) * 100,
        len(X_test),
        test_size * 100,
    )
    return PreprocessResult(X_train, X_test, y_train, y_test)


def run_preprocessing(path: Path | str = DATA_PATH) -> PreprocessResult:
    """Executa o pipeline completo de pré-processamento.

    Args:
        path: Caminho do CSV bruto.

    Returns:
        :class:`PreprocessResult` com os conjuntos de treino e teste.
    """
    raw = load_raw_data(path)
    clean = clean_and_transform(raw)

    removed = [c for c in DROP_COLUMNS if c not in clean.columns]
    logger.info(
        "Verificação: %d/%d colunas descartadas ausentes; %s presente: %s",
        len(removed),
        len(DROP_COLUMNS),
        NEW_FEATURE,
        NEW_FEATURE in clean.columns,
    )

    log_class_distribution(clean[TARGET], "Dataset limpo")
    result = split_dataset(clean)

    logger.info("Features finais (%d): %s", len(result.feature_names), result.feature_names)

    train_dist = log_class_distribution(result.y_train, "Treino")
    test_dist = log_class_distribution(result.y_test, "Teste")
    max_diff = (train_dist["proportion"] - test_dist["proportion"]).abs().max()
    logger.info("Maior diferença de proporção entre treino e teste: %.4f p.p.", max_diff * 100)

    return result


def _configure_logging() -> None:
    """Configura o log do console para a execução direta do módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    try:
        run_preprocessing()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Falha no pré-processamento: %s", exc)
        raise SystemExit(1) from exc
