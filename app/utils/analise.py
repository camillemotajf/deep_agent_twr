import pandas as pd
import json
import ast
from typing import List
from langchain_core.tools import tool # Assumindo que você usa LangChain

# =====================================================
# 1. PARSER DE DICIONÁRIOS (Blindado contra o Mongo)
# =====================================================
def parse_dict_col(val):
    """
    Converte strings JSON, listas do Mongo ou dicionários nativos em dict plano.
    """
    if pd.isna(val): 
        return {}
    
    if isinstance(val, str):
        try: 
            val = json.loads(val)
        except Exception:
            try: 
                val = ast.literal_eval(val)
            except Exception: 
                return {}
                
    if isinstance(val, dict): 
        return val
        
    if isinstance(val, list):
        dict_limpo = {}
        for item in val:
            if isinstance(item, dict):
                if 'name' in item and 'value' in item:
                    dict_limpo[str(item['name'])] = str(item['value'])
                else:
                    for k, v in item.items(): 
                        dict_limpo[str(k)] = str(v)
        return dict_limpo
        
    return {}

# =====================================================
# 2. EXTRATOR PARA FORMATO LONGO (Feature Extraction)
# =====================================================
def extract_features_to_long(
    df: pd.DataFrame, 
    source: str, 
    is_ml: bool = False, 
    class_column: str = "decision"
) -> pd.DataFrame:
    """
    Extrai headers e params diretamente para o formato longo.
    Garante que a classe seja sempre 'bots' ou 'unsafe'.
    """
    records = []
    
    for _, row in df.iterrows():
        # Definindo a classe baseado se é predição (ML) ou label real (Banco)
        if is_ml:
            pred = row.get("pred", -1)
            cls = "bots" if pred == 1 else "unsafe"
        else:
            cls = str(row.get(class_column, "unknown")).lower().strip()
            if cls == "bot": 
                cls = "bots"

        # Extraindo HEADERS
        headers = parse_dict_col(row.get("headers", {}))
        for k, v in headers.items():
            records.append([source, "header", cls, str(k), str(v)])

        # Extraindo PARAMS
        params = parse_dict_col(row.get("request", {}))
        for k, v in params.items():
            records.append([source, "param", cls, str(k), str(v)])

    colunas = ["source", "feature_type", "class", "key", "value"]
    return pd.DataFrame(records, columns=colunas)

# =====================================================
# 3. MOTOR DE MATEMÁTICA (Contagem e Probabilidade)
# =====================================================
def compute_frequencies(df_long: pd.DataFrame, class_totals: dict) -> pd.DataFrame:
    if df_long.empty: 
        return pd.DataFrame()
        
    counts = (
        df_long
        .groupby(["source", "feature_type", "key", "value", "class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["bots", "unsafe"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(columns={"bots": "count_bots", "unsafe": "count_unsafe"})

    # Função auxiliar para calcular a % relativa APENAS à classe
    def get_pct(row, cls):
        total_class_requests = class_totals.get(row["source"], {}).get(cls, 0)
        if total_class_requests == 0:
            return 0.0
        return (row[f"count_{cls}"] / total_class_requests) * 100

    counts["pct_bots"] = counts.apply(lambda r: get_pct(r, "bots"), axis=1)
    counts["pct_unsafe"] = counts.apply(lambda r: get_pct(r, "unsafe"), axis=1)

    return counts

# =====================================================
# 4. ORQUESTRADOR INTERNO DE DATA SCIENCE
# =====================================================
def analyze_frequencies(df_analysis: pd.DataFrame, df_database: pd.DataFrame) -> pd.DataFrame:
    """
    Processa e junta tudo, passando o total real de requisições por classe para a matemática.
    """
    # 1. Contamos o total real de requisições por classe no ML (baseado na predição)
    ml_bots = len(df_analysis[df_analysis["pred"] == 1])
    ml_unsafe = len(df_analysis[df_analysis["pred"] == 0])

    # 2. Contamos o total real no Banco de Dados (baseado no label original)
    db_bots = df_database["decision"].astype(str).str.lower().str.strip().isin(['bot', 'bots', '1', '1.0']).sum()
    db_unsafe = len(df_database) - db_bots

    class_totals = {
        "ml": {"bots": ml_bots, "unsafe": ml_unsafe},
        "database": {"bots": db_bots, "unsafe": db_unsafe}
    }

    # 3. Extração normal
    ml_long = extract_features_to_long(df_analysis, source="ml", is_ml=True)
    db_long = extract_features_to_long(df_database, source="database", is_ml=False, class_column="decision")

    combined_long = pd.concat([ml_long, db_long], ignore_index=True)
    
    # Passamos os totais para o motor matemático
    return compute_frequencies(combined_long, class_totals)