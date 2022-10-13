
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import generate_sentences

st.sidebar.title("Japanese Sentence Generator")
st.sidebar.write("日本語 GPT-2 モデルを使用して続きの文章を生成します")

prompt = st.text_input("プロンプト", "こんにちは、")
num_of_sentences = st.number_input("生成する文章数", 1, 10, 5)
max_length = st.number_input("最大文字列長", 1, 500, 50)

if st.button("生成！"):
    with st.spinner("生成中..."):
        sentences = generate_sentences(prompt, max_length, num_of_sentences)

        st.subheader("Results")
        for sentence in sentences:
            st.write(sentence)
        
st.sidebar.write("")
st.sidebar.write("")

st.sidebar.caption("Copyright (C) 2022 Atsushi Shirafuji")
