from contextvars import ContextVar
from datetime import datetime
import gc
import os
import pandas as pd
from typing import Optional

# _ctx_mongo_df = ContextVar("mongo_df", default=None)
# _ctx_ml_results = ContextVar("ml_results", default=None)
# _ctx_source = ContextVar("traffic_source", default=None)

class AnalysisContext:
    _df_mongo_results: Optional[pd.DataFrame] = None  
    _df_ml_results: Optional[pd.DataFrame] = None     
    _traffic_source: Optional[str] = None
    _file_loaded: Optional[str] = None

    @classmethod
    def set_mongo_data(cls, df: pd.DataFrame, source: str):
        cls._df_mongo_results = df
        cls._traffic_source = source
        cls._df_ml_results = None
        print(f"DEBUG [Context]: Mongo Data Loaded. Rows: {len(df)}")

    @classmethod
    def set_ml_results_data(cls, df: pd.DataFrame):
        cls._df_ml_results = df
        cls._df_mongo_results = None
        gc.collect()
        print(f"DEBUG [Context]: ML Results Stored. Rows: {len(df)}")

    @classmethod
    def set_traffic_source(cls, traffic_source: str):
        cls._traffic_source = traffic_source

    @classmethod
    def get_data_from_mongo(cls) -> pd.DataFrame:
        if cls._df_mongo_results is None:
            raise ValueError("Context Error: No Mongo data loaded yet.")
        return cls._df_mongo_results

    @classmethod
    def get_data_to_analise(cls) -> pd.DataFrame:
        if cls._df_ml_results is not None:
            return cls._df_ml_results
        raise ValueError("Context Error: ML Inference hasn't run yet.")

    @classmethod
    def get_traffic_source(cls) -> str:
        return cls._traffic_source or "unknown"

    @classmethod
    def get_status(cls) -> str:
        status = []
        if cls._df_mongo_results is not None:
            status.append(f"Mongo Raw: {len(cls._df_mongo_results)}")
        else:
            status.append("Mongo Raw: Empty")
            
        if cls._df_ml_results is not None:
            status.append(f"ML Processed: {len(cls._df_ml_results)} rows")
        else:
            status.append("ML Processed: Pending")
            
        return " | ".join(status)
    
    @classmethod
    def clear_memory(cls):
        cls._df_mongo_results = None
        cls._df_ml_results = None

        gc.collect()