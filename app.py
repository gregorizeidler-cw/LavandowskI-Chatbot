import os
import streamlit as st
import re
from dotenv import load_dotenv
from gpt_utils import get_chatgpt_response
from functions import fetch_user_data

# Carrega variáveis de ambiente
load_dotenv()

st.title("💬 Lavandowski Chatbot - AML Analysis")

query_type = st.radio("Selecione o tipo de consulta:", ["Perguntar ao Chatbot", "Consultar um Usuário"])

if query_type == "Perguntar ao Chatbot":
    st.subheader("Faça uma pergunta sobre qualquer dado")
    user_input = st.text_area("Digite sua pergunta para o Lavandowski:", "")

    if st.button("Perguntar"):
        if user_input.strip():
            try:
                # Detecta o user_id automaticamente
                user_id_match = re.search(r'\b\d{5,}\b', user_input)
                user_id = int(user_id_match.group()) if user_id_match else None

                response = get_chatgpt_response(user_input, user_id=user_id)

                st.subheader("🧠 Resposta do Lavandowski:")
                st.markdown(response)

            except Exception as e:
                st.error(f"❌ Erro ao processar a pergunta: {str(e)}")
        else:
            st.warning("⚠️ Por favor, insira uma pergunta válida.")
