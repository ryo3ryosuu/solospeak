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
    st.title("英文の校正")
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
                    """あなたはネイティブの英語教師です。入力された英語テキストは中級の英語学習者による英作文です。
                    以下の形式に従って、アメリカ人のネイティブによる小慣れた英語で校正ししてください(CEFR C1レベルを目安とします）。
                    ・なるべく同じ単語や表現を使わないようにしてください。
                    ・改善ポイントは5つ以内でリストアップし、最後に覚えるべき表現を1つ以上5つ以内でリストアップしてください
                    ・アドバイスは日本語で出力してください。

                    1. 英語の校正結果
                    2. 校正前の英文の評価(CEFR基準)
                    3. 改善ポイント
                    4. 覚えるべきポイントとその解説 

                    出力形式は以下の通りでお願いします。
                    1. 英語の校正結果：<改行>
                       {校正された英文}
                    2. 校正前の英文の評価(CEFR基準)：<改行>
                       {評価に関する文章}
                    3. 改善ポイント：<改行>
                       {改善ポイントに関する文章}
                    4. 覚えるべきポイントとその解説：<改行>
                       {覚えるべきポイントに関する文章}  
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

    input_text = st.text_area(
        "Input Text",
        height=150,
        placeholder="Enter your English text here..."
    )

    if st.button("Analyze Text", disabled=not api_key):
        if not input_text:
            st.error("Please enter some text to analyze.")
            return
            
        try:
            with st.spinner("Analyzing your text..."):
                template = define_prompt_template()
                result = process_transcript_with_llm(llm, input_text, template)
                result_content = result.content
                st.success("AIがあなたの英語についてアドバイスを出力しました！")
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
