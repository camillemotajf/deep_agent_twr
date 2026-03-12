import json
import os
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from app.services.request_service import RequestService
from app.utils.analise import parse_dict_col
import pandas as pd


class SpikeAnalyzerService:
      def __init__(self, request_service):
            self.request_service: RequestService = request_service

      async def analyze_traffic_spikes(
            self, 
            hashes: List[str], 
            start_date: datetime,
            end_date: datetime,
            fields_to_analyze: Optional[List[str]] = None,
            time_window_minutes: str = '5min',
            z_score_threshold: float = 2.0,  
            min_diff_pct: float = 15.0, # Diferença percentual para denunciar o disfarce
            top_n_headers: int = 5
      ) -> Dict[str, Any]:
            """
            Detecta Ataques Sincronizados: períodos onde há pico simultâneo de 'bots' e 'unsafe'.
            Extrai os 'unsafes' dessa zona e compara com os 'unsafes' da zona de calmaria (baixo bot).
            """
            try:
                  # 1. Busca e prepara os dados
                  raw_data = await self.request_service.fetch_training_sample_by_hashes(
                        hashes=hashes, start=start_date, end=end_date
                  )
                  if not raw_data:
                        return {"status": "error", "message": "Nenhum dado encontrado no período."}

                  df = pd.DataFrame(raw_data)
                  df["headers"] = df["headers"].apply(parse_dict_col)
                  df["request"] = df["request"].apply(parse_dict_col)

                  time_col = "datetime"
                  if time_col not in df.columns or 'decision' not in df.columns:
                        return {"status": "error", "message": "Colunas de tempo ou 'decision' ausentes."}

                  df[time_col] = pd.to_datetime(df[time_col])
                  df.set_index(time_col, inplace=True)
                  df.sort_index(inplace=True)

                  # 2. AGRUPAMENTO POR JANELA E DECISÃO
                  df_grouped = df.groupby([pd.Grouper(freq=time_window_minutes), 'decision']).size().unstack(fill_value=0)
                  
                  # Garante que as colunas existem
                  bots_vol = df_grouped['bots'] if 'bots' in df_grouped.columns else pd.Series(0, index=df_grouped.index)
                  unsafe_vol = df_grouped['unsafe'] if 'unsafe' in df_grouped.columns else pd.Series(0, index=df_grouped.index)

                  # 3. MATEMÁTICA DOS PICOS SINCRONIZADOS E DA BASELINE
                  bot_mean, bot_std = bots_vol.mean(), bots_vol.std()
                  unsafe_mean, unsafe_std = unsafe_vol.mean(), unsafe_vol.std()

                  bot_threshold = bot_mean + (z_score_threshold * bot_std)
                  unsafe_threshold = unsafe_mean + (z_score_threshold * unsafe_std)

                  # ZONA 1: Ataque Sincronizado (Pico de Bots AND Pico de Unsafe)
                  is_joint_spike = (bots_vol > bot_threshold) & (unsafe_vol > unsafe_threshold)
                  joint_spike_windows = df_grouped[is_joint_spike].index

                  # ZONA 2: Baseline / Calmaria (Volume de bots abaixo da média)
                  is_baseline = (bots_vol <= bot_mean)
                  baseline_windows = df_grouped[is_baseline].index

                  if len(joint_spike_windows) == 0:
                        return {"status": "success", "message": "Nenhum pico sincronizado detectado."}

                  # 4. ISOLANDO OS 'UNSAFES' NAS DUAS ZONAS
                  df['window'] = df.index.floor(time_window_minutes)
                  unsafe_df = df[df['decision'] == 'unsafe']
                  
                  spike_unsafes = unsafe_df[unsafe_df['window'].isin(joint_spike_windows)]
                  baseline_unsafes = unsafe_df[unsafe_df['window'].isin(baseline_windows)]

                  # 5. EXTRAÇÃO DE DNA (FREQUÊNCIAS)
                  if not fields_to_analyze:
                        fields_to_analyze = ['user-agent', 'ip_api_isp', 'sec-fetch-dest', 'sec-fetch-site']
                  
                  def get_frequencies(sub_df) -> Dict[str, Dict[str, float]]:
                        freqs = {field: {} for field in fields_to_analyze}
                        total_reqs = len(sub_df)
                        if total_reqs == 0: return freqs
                  
                        for field in fields_to_analyze:
                              field_lower = field.lower()
                              extracted_vals = []
                              
                              df_cols_lower = {str(c).lower(): c for c in sub_df.columns}
                              if field_lower in df_cols_lower:
                                    real_col = df_cols_lower[field_lower]
                                    extracted_vals = sub_df[real_col].astype(str).str.strip().str.lower().tolist()
                              elif "headers" in sub_df.columns:
                                    for h_dict in sub_df["headers"]:
                                          if isinstance(h_dict, dict):
                                                h_norm = {str(k).lower(): str(v).strip().lower() for k, v in h_dict.items()}
                                                val = h_norm.get(field_lower)
                                                if val: extracted_vals.append(val)
                              
                              if extracted_vals:
                                    val_counts = pd.Series(extracted_vals).value_counts()
                                    percentages = (val_counts / total_reqs) * 100
                                    freqs[field] = percentages.head(top_n_headers).to_dict()
                        return freqs

                  spike_dna = get_frequencies(spike_unsafes)
                  baseline_dna = get_frequencies(baseline_unsafes)

                  print("Spike DNA: ", json.dumps(spike_dna, indent=2))
                  print("Baseline DNA: ", json.dumps(baseline_dna, indent=2))

                  # 6. COMPARAÇÃO CRUZADA (ACHAR OS CAMUFLADOS DO PICO)
                  report = {
                        "status": "analysis_complete",
                        "joint_spikes_found": len(joint_spike_windows),
                        "joint_spike_timestamps": [str(ts) for ts in joint_spike_windows],
                        "unsafe_reqs_in_joint_spikes": len(spike_unsafes),
                        "unsafe_reqs_in_baseline": len(baseline_unsafes),
                        "camouflaged_bots_detected": {}
                  }

                  for field in fields_to_analyze:
                        suspects = []
                        for val, spike_pct in spike_dna[field].items():
                              if val in ["none", "nan", "unknown", ""]: continue
                              
                              # Pega a representação desse valor na hora da calmaria
                              baseline_pct = baseline_dna[field].get(val, 0.0)
                              
                              # Se o valor explodiu durante o pico mas não existia na calmaria, bingo!
                              diff = spike_pct - baseline_pct

                              print(f"🔎 Analisando {field}: {spike_pct:.1f}% (Pico) - {baseline_pct:.1f}% (Normal) = Diff: {diff:.1f}% | Alarme em: {min_diff_pct}%")

                              if diff >= min_diff_pct:
                                    suspects.append({
                                          "value": val,
                                          "spike_concentration_pct": round(spike_pct, 1),
                                          "baseline_concentration_pct": round(baseline_pct, 1),
                                          "surge_difference_pct": round(diff, 1)
                                    })
                  
                  if suspects:
                        suspects.sort(key=lambda x: x["surge_difference_pct"], reverse=True)
                        report["camouflaged_bots_detected"][field] = suspects

                  if not report["camouflaged_bots_detected"]:
                        report["message"] = "Tráfego Unsafe manteve a mesma proporção. Sem anomalias de camuflagem."

                  return report

            except Exception as e:
                  return {"status": "fatal_error", "message": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}
      
      
      async def generate_spike_graph(
            self, 
            hashes: List[str], 
            start_date: datetime,
            end_date: datetime,
            time_window_minutes: str = '5min', 
            z_score_threshold: float = 2.0,
            output_filepath: str = "temp_data/joint_spike_analysis.png"
      ) -> str:
            """
            Gera o gráfico de série temporal destacando visualmente as faixas de tempo
            onde ocorreu um Ataque Sincronizado (Pico de Bots + Pico de Unsafe ao mesmo tempo).
            """
            try:
                  raw_data = await self.request_service.fetch_training_sample_by_hashes(
                        hashes=hashes, start=start_date, end=end_date 
                  )
                  if not raw_data: return "Nenhum dado encontrado para gerar o gráfico."

                  df = pd.DataFrame(raw_data)
                  time_col = 'datetime'

                  df[time_col] = pd.to_datetime(df[time_col])
                  df.set_index(time_col, inplace=True)
                  df.sort_index(inplace=True)

                  if 'decision' not in df.columns: df['decision'] = 'unknown'

                  # Agrupamento e Separação das Linhas
                  df_grouped = df.groupby([pd.Grouper(freq=time_window_minutes), 'decision']).size().unstack(fill_value=0)

                  bots_vol = df_grouped['bots'] if 'bots' in df_grouped.columns else pd.Series(0, index=df_grouped.index)
                  unsafe_vol = df_grouped['unsafe'] if 'unsafe' in df_grouped.columns else pd.Series(0, index=df_grouped.index)

                  # Matemática dos Limiares
                  bot_mean, bot_std = bots_vol.mean(), bots_vol.std()
                  unsafe_mean, unsafe_std = unsafe_vol.mean(), unsafe_vol.std()

                  bot_threshold = bot_mean + (z_score_threshold * bot_std)
                  unsafe_threshold = unsafe_mean + (z_score_threshold * unsafe_std)

                  # Máscara do Ataque Sincronizado
                  is_joint_spike = (bots_vol > bot_threshold) & (unsafe_vol > unsafe_threshold)
                  joint_spike_windows = df_grouped[is_joint_spike].index

                  # ==========================================
                  # DESENHANDO O GRÁFICO
                  # ==========================================
                  plt.figure(figsize=(15, 7))

                  # Linhas principais de tráfego
                  plt.plot(df_grouped.index, unsafe_vol, label='Humanos (Unsafe)', color='#1f77b4', linewidth=2.5)
                  plt.plot(df_grouped.index, bots_vol, label='Bots', color='#d62728', linewidth=2.5)

                  # Linhas pontilhadas indicando o limiar de alarme para cada classe
                  plt.axhline(y=unsafe_threshold, color='#1f77b4', linestyle=':', alpha=0.6, label='Limiar Alarme Unsafe')
                  plt.axhline(y=bot_threshold, color='#d62728', linestyle=':', alpha=0.6, label='Limiar Alarme Bot')

                  # O DESTAQUE: Faixa Vermelha apenas onde houve o Ataque Sincronizado
                  for i, spike_time in enumerate(joint_spike_windows):
                        end_time = spike_time + pd.Timedelta(time_window_minutes)
                        label = 'Ataque Sincronizado (Camuflagem)' if i == 0 else ""
                        plt.axvspan(spike_time, end_time, color='red', alpha=0.25, label=label)

                  plt.title('Detecção de Ataques Sincronizados (Shadow Bots)', fontsize=16, fontweight='bold', pad=15)
                  plt.xlabel('Linha do Tempo', fontsize=12)
                  plt.ylabel(f'Volume de Requisições ({time_window_minutes})', fontsize=12)

                  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                  plt.xticks(rotation=45)

                  plt.legend(loc='upper right', framealpha=0.9)
                  plt.grid(True, linestyle='--', alpha=0.5)
                  plt.tight_layout()

                  os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                  plt.savefig(output_filepath, dpi=150)
                  plt.close()

                  return output_filepath

            except Exception as e:
                  return f"Erro ao gerar gráfico: {str(e)}\n{traceback.format_exc()}"
            

            