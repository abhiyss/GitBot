import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
#from gitbot.src.chatbot import invoke_chatbot
from chatbot import invoke_chatbot
import boto3
import pickle
#from gitbot.utils import get_indexed_agents
from utils import get_indexed_agents


def chat_interface():
##    with open("./gitbot/streamlit/custom.css") as css:
    with open("./streamlit/custom.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
    current_agent = None
    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    with col1:
        #st.image("./gitbot/pages/L-Rocket-RGB.png", width=250)
        st.image("./streamlit/Logo_gitbot1.png", width=250)
    with col2:
        st.header("GitBot")
    with col3:
        s3 = boto3.client("s3")
        if "agents" not in st.session_state:
            agents = get_indexed_agents()
            agents.sort()
            st.session_state["agents"] = agents
        agent_name = st.selectbox("agent Name", st.session_state["agents"])

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state or (
        "current_agent" in st.session_state
        and st.session_state["current_agent"] != agent_name
    ):
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state or (
        "current_agent" in st.session_state
        and st.session_state["current_agent"] != agent_name
    ):
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state or (
        "current_agent" in st.session_state
        and st.session_state["current_agent"] != agent_name
    ):
        st.session_state["chat_history"] = []
    else:
        for i, j in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)
    if (
        "current_agent" not in st.session_state
        or st.session_state["current_agent"] != agent_name
    ):
        st.session_state["current_agent"] = agent_name

    if prompt:
        message1 = st.chat_message("user")
        message1.write(prompt)

        with st.spinner("Thinking"):
            output = invoke_chatbot(
                agent_name, prompt, st.session_state["chat_history"]
            )
            # Storing the questions, answers and chat history

            st.session_state["chat_answers_history"].append(output["answer"])
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output["answer"]))

        message2 = st.chat_message("assistant")
        message2.write(st.session_state["chat_answers_history"][-1])


chat_interface()
