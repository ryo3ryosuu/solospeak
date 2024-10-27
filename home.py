import streamlit as st

with st.sidebar:
    st.page_link("home.py", label="ホーム", icon="🏠")
    st.page_link("pages/coach.py", label="独り言英語コーチングアプリ", icon="🗣️")
    st.page_link("pages/proofread.py", label="英文の校正", icon="📝")

st.title("Welcome to Home")
st.write("This is the main page of our multi-page app")