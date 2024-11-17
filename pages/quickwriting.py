import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate
from openai import OpenAI
import json

# ページを初期化
def init_page():
    st.title("Quick Writing")
    st.sidebar.title("Options")


# 環境変数からAPIキーを取得
def load_api_key():
    load_dotenv()
    return os.environ['OPENAI_API_KEY']

# OpenAIクライアントを初期化
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# モデルを選択
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4o", "GPT-4o-mini"))
    model_name = "gpt-4o" if model == "GPT-4o" else "gpt-4o-mini"
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.01
    )
    return ChatOpenAI(temperature=temperature, model_name=model_name)

# プロンプトテンプレートを定義
def define_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """あなたは英語の教師です。英作文の課題を生徒に与えてください。

                    課題を3つ出してください。生徒はそのうちの一つを選択します。

                    例えば以下のようなテーマです

                    - Music that is special to you
                    - What foreign country would you like to visit?
                    - The impact of technology on daily life
                    - A story from your childhood 
                    """
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

def process_transcript_with_llm(llm, transcript, template):
    result = llm(template.format_messages(text=transcript))
    return result

def main():
    # APIキーを読み込む
    api_key = load_api_key()
    client = initialize_openai_client(api_key)

    # ページを初期化
    init_page()    

    # ホームへのリンクを追加
    st.markdown("""
                [← Back to Home](/)
                """)

    # LLMを選択
    llm = select_model()

    if st.button("Get Your Homework", disabled=not api_key):
        try:
            with st.spinner("Getting Your Homework..."):
                template = define_prompt_template()
                result = process_transcript_with_llm(llm, "", template)
                result_content = result.content
                st.success("AIがあなたの課題を出力しました！")
                st.write("Content:")
                st.write(result_content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details for debugging:")
            st.error(e)
            
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start.")

if __name__ == "__main__":
    main()
