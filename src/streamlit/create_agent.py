import streamlit as st
#from gitbot.models import EmbeddingModelDisplayNames, LLMModelDisplayNames
#from gitbot.utils import convert_list_to_str, convert_str_to_list
from models import EmbeddingModelDisplayNames, LLMModelDisplayNames
from utils import convert_list_to_str, convert_str_to_list

#from gitbot.streamlit.utils import layout
from streamlit.utils import layout


def vector_embedding_creation_page():
    with open("./gitbot/streamlit/custom.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
    col1, col2, _ = st.columns(3, vertical_alignment="center")
    with col1:
        st.image("./gitbot/pages/L-Rocket-RGB.png", width=250)
    with col2:
        st.subheader("GitHub ChatBot")
    st.subheader("Create a new agent")
    layout()


vector_embedding_creation_page()
