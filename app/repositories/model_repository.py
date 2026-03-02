import boto3
import os
from datetime import datetime
from botocore.exceptions import ClientError
import pytz
from app.config.settings import settings


class ModelRepository:
      def __init__(self):
            self.bucket_name = os.getenv("MODEL_BUCKET_NAME")
            self.region = os.getenv("AWS_REGION")
            self.s3_client = boto3.client('s3', region_name=self.region)

      def _get_s3_prefix(self, is_base_embedding: bool, traffic_source: str = None, emb_config: str = None, env: str = "prod"):

            if is_base_embedding:
                  return f"MIL/embedding"
            if not traffic_source or not emb_config:
                 raise ValueError("MIL models needs 'traffic_source' and 'emb_config'")
            
            return f"MIL/{traffic_source.lower()}/{emb_config.lower()}/{env}/"
            

      def sync_model(self, local_path: str, is_base_embedding: bool = False, traffic_source: str = None, emb_config: str = None) -> bool:
            prefix_prod = self._get_s3_prefix(traffic_source=traffic_source, is_base_embedding=is_base_embedding, emb_config=emb_config, env="prod")

            try:
                  response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix_prod)
                  
                  if "Contents" not in response:
                        print("Not valid archives in s3")
                        return False

                  raw_contents = response["Contents"]

                  # ==========================================
                  # LÓGICA 1: MODELOS DE EMBEDDING (Vários arquivos)
                  # ==========================================
                  if is_base_embedding:
                        base_model_name = os.path.basename(local_path)
                        local_dir = os.path.dirname(local_path)

                        model_files = [
                              obj for obj in raw_contents 
                              if not obj['Key'].endswith('/') and obj['Size'] > 0 and base_model_name in obj['Key']
                        ]

                        if not model_files:
                              print(f"Nenhum arquivo encontrado no S3 para o modelo: {base_model_name}")
                              return False

                        os.makedirs(local_dir, exist_ok=True)

                        for s3_obj in model_files:
                              s3_key = s3_obj['Key']
                              s3_last_modified = s3_obj['LastModified']
                              s3_filename = os.path.basename(s3_key)
                              specific_local_path = os.path.join(local_dir, s3_filename)

                              needs_download = True

                              if os.path.exists(specific_local_path):
                                    local_timestamp = os.path.getmtime(specific_local_path)
                                    local_last_modified = datetime.fromtimestamp(local_timestamp, tz=pytz.utc)

                                    if local_last_modified >= s3_last_modified:
                                          print(f"Local file already updated: {s3_filename}")
                                          needs_download = False

                              if needs_download:
                                    print(f"Updating model file from s3: {s3_key} -> {specific_local_path}")
                                    self.s3_client.download_file(self.bucket_name, s3_key, specific_local_path)

                        return True

                  # ==========================================
                  # LÓGICA 2: MODELOS COMUNS (Apenas o mais recente)
                  # ==========================================
                  else:
                        valid_files = [
                              obj for obj in raw_contents 
                              if not obj['Key'].endswith('/') and obj['Size'] > 0
                        ]

                        if not valid_files:
                              print("Not valid archives in s3")
                              return False

                        # Pega o arquivo mais recente
                        latest_obj = sorted(valid_files, key=lambda x: x['LastModified'], reverse=True)[0]
                        s3_key = latest_obj['Key']
                        s3_last_modified = latest_obj['LastModified']
                        print("Modelo last modified no s3: ", s3_last_modified)

                        if os.path.exists(local_path):
                              local_timestamp = os.path.getmtime(local_path)
                              local_last_modified = datetime.fromtimestamp(local_timestamp, tz=pytz.utc)

                              if local_last_modified >= s3_last_modified:
                                    print("Local file already updated")
                                    return True
                        
                        # Cria a pasta e baixa o arquivo único
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                        print(f"Updating model from s3: {s3_key} -> {local_path}")
                        
                        return True

            except Exception as e:
                  print("Error on sincronizing models from s3", e)
                  return False


      def deploy_model(self, local_model_path: str, is_base_mebedding: bool = False, traffic_source: str = None, emb_config: str = None) -> bool:

            prefix_prod = self._get_s3_prefix(is_base_embedding=is_base_mebedding, traffic_source=traffic_source, emb_config=emb_config, env='prod')
            prefix_archive = self._get_s3_prefix(is_base_embedding=is_base_mebedding, traffic_source=traffic_source, emb_config=emb_config, env="archive")

            self._archive_model(prefix_prod, prefix_archive)

            file_ext = os.path.splitext(local_model_path)[1]
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            s3_filename = f"{date_str}{file_ext}"
            s3_key = f"{prefix_prod}{s3_filename}"

            try:
                  self.s3_client.upload_file(local_model_path, self.bucket_name, s3_key)
                  print("Deploy done successfuly")
                  return True
            except Exception as e:
                  print(f"Erro no deploy: {e}")
                  return False
    

      def delete_local_file(self, local_path: str):

            try:
                  if os.path.exists(local_path):
                        os.remove(local_path)
                        print(f"Modelo local deletado: {local_path}")
                  else:
                        print(f"Arquivo não existe para deletar: {local_path}")
            except Exception as e:
                  print(f"Erro ao deletar arquivo: {e}")


      
      def _archive_model(self, prefix_prod, prefix_archive):
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix_prod)
            if 'Contents' not in response:
                return
            
            for obj in response['Contents']:
                old_key = obj['Key']
                filename_only = old_key.split('/')[-1]

                if not filename_only: continue

                archive_key = f"{prefix_archive}{filename_only}"
                
                self.s3_client.copy_object(
                    Bucket=self.bucket_name,
                    CopySource={'Bucket': self.bucket_name, 'Key': old_key},
                    Key=archive_key
                )
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=old_key)
        except ClientError as e:
            print(f"Aviso ao arquivar: {e}")

      def delete_local_file(self, local_path: str):
            try:
                  if os.path.exists(local_path):
                        os.remove(local_path)
                  print(f"Modelo local deletado: {local_path}")
            except Exception as e:
                  print(f"Erro ao deletar arquivo: {e}")



