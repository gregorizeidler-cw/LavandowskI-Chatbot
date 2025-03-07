import os
from dotenv import load_dotenv
from openai import OpenAI

# Carrega as variáveis de ambiente
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "Você tem acesso a todas as tabelas de clientes, transações e relacionamentos financeiros. "
    "Quando um usuário perguntar sobre um cliente, utilize os dados fornecidos antes de responder. "
    "Se houver registros, responda com base nas informações encontradas. "
    "Se não houver registros, responda 'Não há informações disponíveis'."
)

def get_chatgpt_response(prompt, model="gpt-4o-2024-11-20", is_analysis=False, user_id=None):
    """
    Envia um prompt para o modelo GPT especificado e retorna a resposta.

    Args:
        prompt (str): A pergunta do usuário.
        model (str): O modelo GPT a ser utilizado.
        is_analysis (bool): Define se a pergunta é uma análise detalhada ou resposta direta.
        user_id (int, optional): ID do usuário para buscar dados.

    Returns:
        str: Resposta do modelo ou erro.
    """
    try:
        from functions import fetch_user_data  

        user_data_prompt = ""
        if user_id:
            df = fetch_user_data(user_id)

            if not df.empty:
                user_data_prompt = f"\n\n📊 **Dados do Cliente {user_id}:**\n{df.to_string(index=False)}"
                print(f"📤 Enviando para o GPT:\n{user_data_prompt}")
            else:
                user_data_prompt = f"\n\n🚫 Nenhuma informação encontrada para o cliente {user_id}."

        print(f"📥 Prompt enviado para o GPT:\n{prompt}\n{user_data_prompt}")

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + user_data_prompt},
            ]
        }

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Erro: {str(e)}"
