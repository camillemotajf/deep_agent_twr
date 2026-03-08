from typing import List, Dict
from user_agents import parse
from langchain.tools import tool
import shodan
from app.config.settings import settings
import requests
import json
import pandas as pd
import os
from app.utils.analise import parse_dict_col



HEADERS_TO_ANALISE = ["Sec-Ch-Ua,Sec-Ch-Ua-Full-Version","Sec-Ch-Ua-Platform", "Sec-Ch-Ua-Platform-Version","Sec-Ch-Ua-Arch","Sec-Ch-Bitness","Sec-Ch-Ua-Model", "Sec-Ch-Ua-Mobile","Device-Memor", "Dpr","Viewport-Width","Downlink", "Ect","Rtt","Save-Data", "Sec-Ch-Prefers-Color-Scheme","Sec-Ch-Prefers-Reduced-Motion", "Sec-Ch-Prefers-Contrast","Sec-Ch-Prefers-Reduced-Data", "Sec-Ch-Forced-Colors"]
                      
shodan_client = shodan.Shodan(settings.SHODAN_API_KEY)

@tool
def analisar_ip_shodan(ip_addresses: List[str]) -> dict:
    """
    Consulta a API do Shodan para um endereço IP específico.
    Retorna o ISP, a Organização, as portas abertas e tags de segurança (ex: 'vpn', 'proxy', 'tor').
    Ideal para descobrir se um IP é um servidor em um Data Center ou um nó malicioso.
    """
    final_results = {}
        # Consulta o IP no Shodan
    for ip in ip_addresses:
        try:
            host_info = shodan_client.host(ip)

            info_ip = {
                "found_in_shodan": True,
                "isp": host_info.get("isp", "Unknow"),
                "organizacao": host_info.get("org", "Unknow"),
                "asn": host_info.get("asn", "Unknow"),
                "portas_abertas": host_info.get("ports", []),
                "tags": host_info.get("tags", []), 
                "risco": "HIGH"
            }

        except shodan.APIError as e:
            # Se o erro for "No information available for that IP"
            if "No information available" in str(e):
                info_ip = {
                    "encontrado_no_shodan": False,
                    "motivo": "IP não listado. Provavelmente um dispositivo residencial/mobile comum.",
                    "risco": "BAIXO"
                }
            else:
                info_ip = {"erro_api": str(e)}
        
            final_results[ip] = info_ip

        return {
            "status": "results found",
            "results": final_results
        }

from typing import List, Dict

def format_api_post(headers: List[Dict]) -> Dict:
    """
    Extrai apenas headers presentes em HEADERS_TO_ANALISE
    e formata no padrão esperado pela API.
    """
    results = []

    for header_dict in headers:
        if not isinstance(header_dict, dict):
            continue

        for key, value in header_dict.items():

            # ignora valores vazios
            if value is None:
                continue

            # verifica se o header está na lista
            if any(h.lower() == key.lower() for h in HEADERS_TO_ANALISE):

                results.append({
                    "name": key.upper(),
                    "value": str(value)
                })

    return {"headers": results}

@tool
def analisar_user_agent(headers: List[Dict]) -> dict:
    """
    Analisa uma string bruta de User-Agent HTTP.
    Retorna um dicionário com o navegador, sistema operativo, tipo de dispositivo e se é um bot conhecido.
    
    """
    req_headers = {
        'X-API-KEY': settings.API_BROWSER_KEY,
    }

    post_data = format_api_post(headers=headers)

    try:
        response = requests.post(
            settings.API_BROWSER_URL,
            json=post_data,
            headers=req_headers,
            timeout=10
        )

        if response.status_code == 404:
            return {
                "status": "api_error",
                "error": f"Endpoint não encontrado (404): {settings.API_BROWSER_URL}"
            }

        if response.status_code != 200:
            return {
                "status": "api_error",
                "error": f"HTTP {response.status_code}: {response.text}"
            }

        result = response.json()

        if result.get("error"):
            return {
                "status": "api_browser_error",
                "message": result.get("error")
            }

        # ✅ FALTAVA ISSO
        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        return {
            "status": "process_error",
            "message": str(e)
        }
# @tool
# def analisar_user_agent(headers: List[Dict]) -> dict:

#     req_headers = {
#         'X-API-KEY': settings.API_BROWSER_KEY,
#     }

#     post_data = format_api_post(headers=headers)
#     print("Post data: ", post_data)

#     try: 
#         response = requests.post(
#             settings.API_BROWSER_URL,
#             json=post_data,  # CORRIGIDO
#             headers=req_headers,
#             timeout=10
#         )
#         print(f"Status Code: {response.status_code}")
#         print(f"Response: {response.text}")
#         if response.status_code == 404:
#             return {
#                 "status": "api_error",
#                 "error": f"Endpoint não encontrado (404): {settings.API_BROWSER_URL}",
#                 "suggestion": "Verifique se API_BROWSER_URL está correto nas settings"
#             }
#         if response.status_code != 200:
#             return {
#                 "status": "api_error",
#                 "error": f"HTTP {response.status_code}: {response.text}"
#             }
#         result = response.json()
#         if result.get("error", None):
#             return {
#                 "status": "api browser error",
#                 "message": result.get("error")
#             }


#     except Exception as e:
#         return {
#             "status": "process error",
#             "message": e
#         }
        

@tool
def enrich_parquet_data(filepath: str) -> str:
    """
    Reads a Parquet file containing HTTP requests, analyzes headers using the Browser API,
    and saves an enriched parquet file with the API response.
    """
    if not os.path.exists(filepath):
        return f"Error: File {filepath} not found."

    try:
        df = pd.read_parquet(filepath)

        # garantir que headers são dict
        df["headers"] = df["headers"].apply(parse_dict_col)

        print("Dados depois do parse:\n", df["headers"].head())

        browser_results = []

        for headers in df["headers"]:

            if not isinstance(headers, dict):
                browser_results.append(None)
                continue

            try:
                # transformar dict -> lista de dicts (formato esperado)
                headers_list = [headers]

                ua_report = analisar_user_agent.func(headers_list)

                browser_results.append(ua_report)

            except Exception as e:
                browser_results.append({
                    "status": "process_error",
                    "message": str(e)
                })

        # adicionar coluna com resposta da API
        df["api_browser_response"] = browser_results

        # salvar parquet enriquecido
        enriched_filepath = filepath.replace("raw_", "enriched_")
        df.to_parquet(enriched_filepath)

        return (
            f"SUCCESS: Data enriched and saved to {enriched_filepath}. "
            f"Processed {len(df)} requests with Browser API."
        )

    except Exception as e:
        return f"Error enriching data: {str(e)}"