import datetime
import pandas as pd
from google.cloud import bigquery
import json
import decimal
import logging
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do BigQuery
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("LOCATION")
client = bigquery.Client(project=project_id, location=location)

# Configuração de logs
logging.basicConfig(level=logging.ERROR)

# Formatação de JSON para lidar com valores Decimais
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()
        else:
            return super().default(obj)

# Função genérica para executar queries
def execute_query(query):
    """Executa uma consulta SQL no BigQuery e retorna um DataFrame."""
    try:
        df = client.query(query).result().to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Erro ao executar a query: {e}")
        return pd.DataFrame()

# Tabelas que devem ser consultadas no projeto
tables = {
    "lawsuit_data": "infinitepay-production.metrics_amlft.lavandowski_lawsuits_data",
    "business_relationships": "infinitepay-production.metrics_amlft.lavandowski_business_relationships_data",
    "sanctions_history": "infinitepay-production.metrics_amlft.sanctions_history",
    "denied_transactions": "infinitepay-production.metrics_amlft.lavandowski_risk_transactions_data",
    "denied_pix_transactions": "infinitepay-production.metrics_amlft.lavandowski_risk_pix_transfers_data",
    "prison_transactions": "infinitepay-production.metrics_amlft.prison_transactions",
    "merchant_report": "metrics_amlft.merchant_report",
    "issuing_payments": "infinitepay-production.metrics_amlft.lavandowski_issuing_payments_data",
    "pix_concentration": "metrics_amlft.pix_concentration",
    "offense_analysis": "infinitepay-production.metrics_amlft.lavandowski_offense_analysis_data",
    "phonecast": "infinitepay-production.metrics_amlft.lavandowski_phonecast_data",
    "user_device": "metrics_amlft.user_device",
    "cardholder_report": "metrics_amlft.cardholder_report",
    "online_store_data": "infinitepay-production.metrics_amlft.lavandowski_online_store_data",
    "cardholder_concentration": "metrics_amlft.cardholder_concentration",
    "issuing_concentration": "metrics_amlft.issuing_concentration",
    "bank_slips_alert": "metrics_amlft.bank_slips_alert",
    "gafi_alert": "metrics_amlft.gafi_alert",
    "government_corporate_cards": "metrics_amlft.government_corporate_cards",
    "international_cards": "metrics_amlft.international_cards",
    "betting_houses_alert": "metrics_amlft.betting_houses_alert",
    "ch_alert": "metrics_amlft.ch_alert",
    "pep_pix_alert": "metrics_amlft.pep_pix_alert",
    "issuing_transactions": "metrics_amlft.issuing_transactions"
}

# Possíveis colunas de identificação do usuário nas tabelas
user_columns = ["user_id", "customer_id", "client_id", "debit_party", "merchant_id", "account_id"]

def fetch_user_data(user_id: int) -> pd.DataFrame:
    """
    Busca os dados do usuário em todas as tabelas relevantes.
    
    Args:
        user_id (int): ID do usuário a ser consultado.
    
    Returns:
        pd.DataFrame: Dados combinados de todas as tabelas.
    """
    combined_data = []
    for label, table in tables.items():
        try:
            # Identifica a coluna correta
            column_check_query = f"""
            SELECT column_name FROM `{table[:-table[::-1].index(".") - 1]}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table.split(".")[-1]}'
            """
            columns_df = execute_query(column_check_query)
            available_columns = set(columns_df["column_name"].tolist())

            user_column = next((col for col in user_columns if col in available_columns), None)

            if not user_column:
                logging.warning(f"⚠️ A tabela {table} NÃO contém colunas identificáveis para busca de usuário.")
                continue

            # Executa a consulta com a coluna correta
            query = f"SELECT * FROM `{table}` WHERE {user_column} = {user_id} LIMIT 100"
            df = execute_query(query)
            if not df.empty:
                df["source_table"] = table  # Indica a origem dos dados
                combined_data.append(df)

        except Exception as e:
            logging.error(f"❌ Erro ao buscar dados na tabela {table}: {e}")

    return pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()

# Exemplo de uso
if __name__ == "__main__":
    user_id = 123456789  # Substituir por um ID real
    user_data = fetch_user_data(user_id)
    print(user_data)
