import streamlit as st
#from gitbot.models import EmbeddingModelDisplayNames, LLMModelDisplayNames
from models import EmbeddingModelDisplayNames, LLMModelDisplayNames

import boto3
import pickle
import json
#from gitbot.utils import convert_list_to_str, convert_str_to_list
#from gitbot.streamlit.utils import layout
#from gitbot.utils import (
#    get_indexed_agents,
#    save_indexed_agents,
#    delete_s3_agent_contents,
#)
from utils1 import convert_list_to_str, convert_str_to_list
from utils import layout
from utils1 import (
    get_indexed_agents,
    save_indexed_agents,
    delete_s3_agent_contents,
)
import time
#from gitbot.main import agentMessage, streamlit_agent_update_endpoint
from main import agentMessage, streamlit_agent_update_endpoint


@st.dialog("Rebuid agent")
def rebuild_agent(agent_name):
    st.warning(f"Are you sure you want to rebuild the agent {agent_name}?")
    st.write("This action cannot be undone")
    if st.button("Rebuild"):
        with st.spinner("Working on your agent..."):
            s3 = boto3.client("s3")
            config = (
                s3.get_object(
                    #Bucket="sagemaker-us-east-2-534295958235",
                    Bucket="myprojects-2025",
                    Key=f"gitbot/{agent_name}/config.json",
                )["Body"]
                .read()
                .decode("utf-8")
            )
            config = json.loads(config)
            config["agent_name"] = agent_name
            streamlit_agent_update_endpoint(
                                agentMessage(**config), is_llm_only_update=False
                            )
            
            st.write("agent rebuilt!")

        st.balloons()
        time.sleep(3)

        st.rerun()

@st.dialog("Edit agent")
def edit_agent(agent_name):
    s3 = boto3.client("s3")
    config = (
        s3.get_object(
            #Bucket="sagemaker-us-east-2-534295958235",
            Bucket="myprojects-2025",
            Key=f"gitbot/{agent_name}/config.json",
        )["Body"]
        .read()
        .decode("utf-8")
    )
    config = json.loads(config)
    st.header(agent_name)

    if config["prompt"] == "{query}":
        prompt = None
    else:
        prompt = config["prompt"]
    layout(
        agent_name,
        convert_list_to_str(config["github_repos"]),
        convert_list_to_str(config["include_branches"]),
        config["include_file_types"],
        convert_list_to_str(config["include_folders"]),
        convert_list_to_str(config["exclude_folders"]),
        config["documentation_folder_path"],
        convert_list_to_str(config["exclude_file_types"]),
        config["embedding_model_name"],
        config["llm_model_name"],
        prompt,
    )


@st.dialog("Delete agent")
def delete_agent(agent_name):
    st.warning(f"Are you sure you want to delete the agent {agent_name}?")
    st.write("This action cannot be undone")
    if st.button("Delete"):
        agents = get_indexed_agents()
        if agent_name not in agents:
            st.error("agent not found")

        agents.remove(agent_name)

        save_indexed_agents(agents)

        delete_s3_agent_contents(agent_name)

        st.success("agent deleted successfully")

        agents = get_indexed_agents()
        agents.sort()
        st.session_state["agents"] = agents

        time.sleep(3)

        st.rerun()


def existing_agent_page():
    #with open("./gitbot/streamlit/custom.css") as css:
    with open("./streamlit/custom.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
        
    if "agents" not in st.session_state:
        agents = get_indexed_agents()
        agents.sort()
        st.session_state["agents"] = agents
        
    agents = st.session_state["agents"]
    
    for agent in agents:
        col1, _, col3, col4, col5 = st.columns(5, vertical_alignment="center")
        with col1:
            st.write(agent)
        with col3:
            st.button(
                "Rebuild",
                key="Rebuild agent-{}".format(agent),
                on_click=rebuild_agent,
                args=(agent,),
                use_container_width=True,
            )
        with col4:
            st.button(
                "Edit",
                key="Edit agent-{}".format(agent),
                on_click=edit_agent,
                args=(agent,),
                use_container_width=True,
            )
        with col5:
            st.button(
                "Delete",
                key="Delete agent-{}".format(agent),
                on_click=delete_agent,
                args=(agent,),
                use_container_width=True,
            )


existing_agent_page()
