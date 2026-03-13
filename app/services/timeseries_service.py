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

import plotly.express as px


class DatetimeAnalyzesService:
      def __init__(self, request_service: RequestService, start_date: datetime, end_date: datetime):
            self.request_service: RequestService = request_service
            self.start_date = start_date
            self.end_date = end_date

      async def analyze_traffic_spikes(
            self, 
            hashes: List[str], 
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
                        hashes=hashes, start=self.start_date, end=self.end_date
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
                              
                              baseline_pct = baseline_dna[field].get(val, 0.0)
                              diff = spike_pct - baseline_pct

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
                        hashes=hashes, start=self.start_date, end=self.end_date 
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
            

      async def group_reqs_by_campaign(self, hashes: List[str]):

            dfs_temp = []
            
            for current_hash in hashes:
                  requests = await self.request_service.fetch_training_sample_by_hashes(
                        hashes=[current_hash],
                        start=self.start_date,
                        end=self.end_date
                  )

                  if requests:
                        temp_df = pd.DataFrame(requests)
                        temp_df["hash"] = current_hash 
                        dfs_temp.append(temp_df)
            
            if dfs_temp:
                  df = pd.concat(dfs_temp, ignore_index=True)
            else:
                  df = pd.DataFrame()

            return df
            
            

      async def analise_fingerprint_by_campaign(self, hash: str, headers_list: list, bucket_size: str = '10min'):
            
            requests = await self.request_service.fetch_training_sample_by_hashes(
                  hashes=[hash],
                  start=self.start_date,
                  end=self.end_date
            )

            df = pd.DataFrame(requests)
            df["headers"] = df["headers"].apply(parse_dict_col)
            df["request"] = df["request"].apply(parse_dict_col)

            print("Tipos de headers: ", type(df["headers"].iloc[0]))


            if df.empty or "ip" not in df.columns or "headers" not in df.columns:
                  return {"status": "Dados insuficientes (IP ou Headers ausentes)."}
            
            for header in headers_list:
                  df[header] = df["headers"].apply(
                        lambda x: str(x.get(header, "None")) if isinstance(x, dict) else "Invalid"
                  )
            
         
            cols_to_concat = ['ip'] + headers_list
            df['fingerprint'] = df[cols_to_concat].astype(str).agg(' | '.join, axis=1)
            
            time_col = "datetime"
            df_grouped = df.groupby([
                  pd.Grouper(key=time_col, freq=bucket_size), # Eixo X
                  'fingerprint',                              # Eixo Y (Assinatura única)
                  'decision'                                  # Cor
            ]).size().reset_index(name='request_count')

            df_grouped['ip'] = df_grouped['fingerprint'].apply(lambda x: x.split(' | ')[0])
            
            return df_grouped
      
      async def campaigns_analyzes(self, hashes: List[str], bucket_size: str = "1min"):

            df = self.group_reqs_by_campaign(hashes=hashes)
            
            if df.empty or "ip" not in df.columns or "headers" not in df.columns:
                  return {"status": "Insuficient data or hash columns not found."}
                        
            df["headers"] = df["headers"].apply(parse_dict_col)

            time_col = "datetime"
            df_grouped = df.groupby([
                  pd.Grouper(key=time_col, freq=bucket_size),
                  "hash",
                  "decision"
            ]).size().reset_index(name="request_count")

            return df_grouped
      

      @staticmethod
      def plot_campaigns_behavior(df_grouped: pd.DataFrame, output_path: str = "comportamento_campanhas.html"):

            # Verifica se o retorno não foi o dicionário de erro ou se está vazio
            if isinstance(df_grouped, dict) or df_grouped.empty:
                  print("Nenhum dado encontrado ou erro na extração. Gráfico não gerado.")
                  return

            # Ordena por tempo para garantir que a linha siga a cronologia correta
            df_plot = df_grouped.sort_values(by="datetime")

            # Cria o gráfico de linhas
            fig = px.line(
                  df_plot,
                  x="datetime",
                  y="request_count",
                  color="hash",             # Cada hash (campanha) terá uma cor diferente
                  line_dash="decision",     # Cada decisão (bot, human, unsafe) terá um tipo de linha (sólida, tracejada...)
                  markers=True,             # Adiciona bolinhas em cada ponto (ajuda a ver picos de 1 min isolados)
                  hover_data={
                        "datetime": "|%d/%m %H:%M:%S",
                        "request_count": ":,.0f",
                        "hash": True,
                        "decision": True
                  },
                  title="Comportamento de Requisições por Campanha ao Longo do Tempo"
            )

            # Ajusta o visual para manter o seu padrão "Dark Mode" hacker/cybersec
            fig.update_layout(
                  template="plotly_dark",
                  height=750,
                  xaxis_title="Janela de Tempo",
                  yaxis_title="Volume de Requisições",
                  legend_title="Campanha (Cor) / Decisão (Linha)",
                  font=dict(
                        family="Courier New, monospace",
                        size=12
                  ),
                  xaxis=dict(
                        tickformat="%H:%M\n%d/%m",
                        gridcolor="rgba(255,255,255,0.15)",
                        showgrid=True
                  ),
                  yaxis=dict(
                        gridcolor="rgba(255,255,255,0.1)",
                  ),
                  paper_bgcolor="#0d1117",
                  plot_bgcolor="#0d1117",
                  hovermode="x unified"  # Mostra uma linha vertical no hover e compara tudo naquele exato minuto
            )

            fig.update_xaxes(
                  showspikes=True,
                  spikemode="across",
                  spikesnap="cursor",
                  spikecolor="white",
                  spikethickness=1
            )

            # Garante a criação da pasta caso o output_path tenha diretórios
            dir_name = os.path.dirname(output_path)
            if dir_name:
                  os.makedirs(dir_name, exist_ok=True)
                  
            fig.write_html(output_path)
            print(f"Gráfico gerado com sucesso em: {output_path}")



      
      async def analyzes_cross_cpg_fp(self, hashes: List[str], headers_list: List[str], bucket_size: str = "1min"):

            df = self.group_reqs_by_campaign(hashes=hashes)
            
            if df.empty or "ip" not in df.columns or "headers" not in df.columns:
                  return {"status": "Insuficient data or hash columns not found."}
                        
            df["headers"] = df["headers"].apply(parse_dict_col)
            
            for header in headers_list: 
                  header_lower = header.lower()
                  df[header] = df["headers"].apply(
                        lambda x: str(
                              {str(k).lower(): v for k, v in x.items()}.get(header_lower, "Absent")
                        ) if isinstance(x, dict) else "Invalid"
                  )

            cols_to_concat = ["ip"] + headers_list
            df["fingerprint"] = df[cols_to_concat].astype(str).agg(" | ".join, axis=1)

            time_col = "datetime"
            df_grouped = df.groupby([
                  pd.Grouper(key=time_col, freq=bucket_size),
                  "fingerprint",
                  "hash",
                  "decision"
            ]).size().reset_index(name="request_count")

            cpgs_per_fp = df_grouped.groupby('fingerprint')['hash'].nunique()
            sus_fps = cpgs_per_fp[cpgs_per_fp > 1].index.to_list()

            df_croos_cpg = df_grouped[df_grouped['fingerprint'].isin(sus_fps)].copy()
            if not df_croos_cpg.empty:
                  df_croos_cpg['ip'] = df_croos_cpg["fingerprint"].apply(lambda x: str(x).split(" | ")[0])

            return df_croos_cpg
      
      @staticmethod
      def plot_cross_campaign_behaviour(df_grouped: pd.DataFrame, traffic_source_name: str, output_path="analise_cross_campaign.html"):

            if isinstance(df_grouped, dict) or df_grouped.empty:
                  print("Nenhum fingerprint cruzado (bot) encontrado ou erro nos dados.")
                  return

            # Pegar os Top 30 Fingerprints suspeitos com maior volume total
            top_fingerprints = (
                  df_grouped.groupby("fingerprint")["request_count"]
                  .sum()
                  .nlargest(30)
                  .index
            )

            df_plot = df_grouped[df_grouped["fingerprint"].isin(top_fingerprints)].copy()

            # Ordenar eixo Y
            total_vol_per_fp = df_plot.groupby("fingerprint")["request_count"].sum().sort_values(ascending=False)
            sorted_fps = total_vol_per_fp.index.tolist()

            # Truncar nomes longos no eixo Y
            def truncate_label(fp_str):
                  return fp_str[:42] + "..." if len(str(fp_str)) > 45 else str(fp_str)
            
            df_plot['display_name'] = df_plot['fingerprint'].apply(truncate_label)
            sorted_display_names = [truncate_label(fp) for fp in sorted_fps]

            # Criar o gráfico
            fig = px.scatter(
                  df_plot,
                  x="datetime",
                  y="display_name",
                  size="request_count",
                  color="hash", # AQUI ESTÁ O SEGREDO: A cor agora é a Campanha (Hash)
                  hover_name="ip",
                  hover_data={
                        "display_name": False,
                        "fingerprint": True,
                        "hash": True, # Mostra qual campanha foi acessada no tooltip
                        "datetime": "|%d/%m %H:%M:%S",
                        "request_count": ":,.0f",
                        "decision": True
                  },
                  title=f"Cross-Campaign Bot Detection - Source: {traffic_source_name} (Top 30 Suspicious)",
                  category_orders={"display_name": sorted_display_names}
            )

            sizeref = 2. * df_plot["request_count"].max() / (40 ** 2)

            fig.update_traces(
                  marker=dict(
                        sizeref=sizeref,
                        sizemin=5,
                        sizemode="area",
                        opacity=0.8,
                        line=dict(width=1, color='white') # Borda branca para destacar bolhas sobrepostas
                  )
            )

            fig.update_layout(
                  template="plotly_dark",
                  height=850,
                  xaxis_title="Time Window",
                  yaxis_title="Suspicious Signature (IP | Headers)",
                  legend_title="Campaign (Hash)", # Título da legenda
                  font=dict(family="Courier New, monospace", size=11),
                  xaxis=dict(tickformat="%H:%M\n%d/%m", gridcolor="rgba(255,255,255,0.15)", showgrid=True),
                  yaxis=dict(gridcolor="rgba(255,255,255,0.1)", dtick=1),
                  paper_bgcolor="#0d1117",
                  plot_bgcolor="#0d1117",
                  hoverlabel=dict(namelength=-1)
            )

            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="white", spikethickness=1)

            # Corrige o problema do diretório
            dir_name = os.path.dirname(output_path)
            if dir_name:
                  os.makedirs(dir_name, exist_ok=True)
                  
            fig.write_html(output_path)
      
            
      async def analise_ips_by_campaign(self, hash: str, bucket_size: str = '10min'):

            requests = await self.request_service.fetch_training_sample_by_hashes(
                  hashes=[hash],
                  start=self.start_date,
                  end=self.end_date
            )

            df = pd.DataFrame(requests)

            if not "ip" in df.columns:
                  return {"status": "ip not found in requests column."}
            
            time_col = "datetime"
            
            df_grouped = df.groupby([
                pd.Grouper(key=time_col, freq=bucket_size), # Agrupa pelo tempo (eixo X)
                'ip',                                       # Agrupa pelo IP (eixo Y)
                'decision'                                  # Mantém a classificação (para cor)
            ]).size().reset_index(name='request_count')

            return df_grouped
      
      @staticmethod
      def plot_ip_behaviour(df_grouped: pd.DataFrame, campaign: str, output_path="analise_ip.html"):

            if df_grouped.empty:
                  return {"erro": "df is empty"}

            # Top IPs
            top_ips = (
                  df_grouped.groupby("ip")["request_count"]
                  .sum()
                  .nlargest(30)
                  .index
            )

            df_grouped = df_grouped[df_grouped["ip"].isin(top_ips)]

            total_vol_per_ip = (
                  df_grouped.groupby("ip")["request_count"]
                  .sum()
                  .sort_values(ascending=False)
            )

            sorted_ips = total_vol_per_ip.index.tolist()

            fig = px.scatter(
                  df_grouped,
                  x="datetime",
                  y="ip",
                  size="request_count",
                  color="decision",
                  hover_name="ip",
                  hover_data={
                        "datetime": "|%d/%m %H:%M:%S",
                        "request_count": ":,.0f",
                        "decision": True
                  },
                  color_discrete_map={
                        "bot": "#ff4d4d",
                        "human": "#2ecc71",
                        "unsafe": "#ffa500"
                  },
                  title=f"IP Activity Timeline – {campaign} (Top 30 IPs)",
                  category_orders={"ip": sorted_ips}
            )

            sizeref = 2. * df_grouped["request_count"].max() / (40 ** 2)

            fig.update_traces(
                  marker=dict(
                        sizeref=sizeref,
                        sizemin=4,
                        sizemode="area",
                        opacity=0.75
                  )
            )

            fig.update_layout(
                  template="plotly_dark",
                  height=800,
                  xaxis_title="Time Window",
                  yaxis_title="Source IP",
                  font=dict(
                        family="Courier New, monospace",
                        size=12
                  ),
                  xaxis=dict(
                        tickformat="%H:%M\n%d/%m",
                        gridcolor="rgba(255,255,255,0.15)",
                        showgrid=True
                  ),
                  yaxis=dict(
                        gridcolor="rgba(255,255,255,0.1)",
                        dtick=1
                  ),
                  paper_bgcolor="#0d1117",
                  plot_bgcolor="#0d1117"
            )

            fig.update_xaxes(
                  showspikes=True,
                  spikemode="across",
                  spikesnap="cursor",
                  spikecolor="white",
                  spikethickness=1
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)

      @staticmethod
      def plot_fingerprint_behaviour(df_grouped: pd.DataFrame, campaign: str, output_path="analise_fingerprint.html"):

            if df_grouped.empty or "erro" in df_grouped:
                  return {"erro": "df is empty or invalid"}

            # Pegar os Top 30 Fingerprints com mais requisições
            top_fingerprints = (
                  df_grouped.groupby("fingerprint")["request_count"]
                  .sum()
                  .nlargest(30)
                  .index
            )

            df_grouped = df_grouped[df_grouped["fingerprint"].isin(top_fingerprints)].copy()

            # Ordenar para o gráfico
            total_vol_per_fp = (
                  df_grouped.groupby("fingerprint")["request_count"]
                  .sum()
                  .sort_values(ascending=False)
            )
            sorted_fps = total_vol_per_fp.index.tolist()

            # TRUNCAMENTO PARA O EIXO Y: Mostra o IP e apenas os primeiros 30 caracteres do resto
            # Assim o eixo Y não fica largo demais.
            def truncate_label(fp_str):
                  if len(fp_str) > 45:
                        return fp_str[:42] + "..."
                  return fp_str
            
            df_grouped['display_name'] = df_grouped['fingerprint'].apply(truncate_label)
            # Precisamos ordenar os display_names na mesma ordem dos fingerprints originais
            sorted_display_names = [truncate_label(fp) for fp in sorted_fps]

            fig = px.scatter(
                  df_grouped,
                  x="datetime",
                  y="display_name", # Usamos o nome truncado no eixo Y
                  size="request_count",
                  color="decision",
                  hover_name="ip", # Destaca o IP no título do tooltip
                  hover_data={
                        "display_name": False, # Esconde o truncado do hover
                        "fingerprint": True,   # Mostra a string GIGANTE e COMPLETA no hover
                        "datetime": "|%d/%m %H:%M:%S",
                        "request_count": ":,.0f",
                        "decision": True
                  },
                  color_discrete_map={
                        "bot": "#ff4d4d",
                        "human": "#2ecc71",
                        "unsafe": "#ffa500"
                  },
                  title=f"Device/User Activity Timeline (IP + Headers) – {campaign} (Top 30)",
                  category_orders={"display_name": sorted_display_names}
            )

            sizeref = 2. * df_grouped["request_count"].max() / (40 ** 2)

            fig.update_traces(
                  marker=dict(
                        sizeref=sizeref,
                        sizemin=4,
                        sizemode="area",
                        opacity=0.75
                  )
            )

            fig.update_layout(
                  template="plotly_dark",
                  height=850, # Um pouco mais alto para acomodar os textos
                  xaxis_title="Time Window",
                  yaxis_title="User Signature (IP | Headers)",
                  font=dict(
                        family="Courier New, monospace",
                        size=11 # Fonte um pouco menor para o eixo Y
                  ),
                  xaxis=dict(
                        tickformat="%H:%M\n%d/%m",
                        gridcolor="rgba(255,255,255,0.15)",
                        showgrid=True
                  ),
                  yaxis=dict(
                        gridcolor="rgba(255,255,255,0.1)",
                        dtick=1
                  ),
                  paper_bgcolor="#0d1117",
                  plot_bgcolor="#0d1117",
                  hoverlabel=dict(
                  namelength=-1 # Permite que o hover mostre o texto completo sem cortar
                  )
            )

            fig.update_xaxes(
                  showspikes=True,
                  spikemode="across",
                  spikesnap="cursor",
                  spikecolor="white",
                  spikethickness=1
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
                        

            