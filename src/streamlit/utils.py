import streamlit as st
#from gitbot.utils import convert_str_to_list, get_indexed_agents
#from gitbot.models import EmbeddingModelDisplayNames, LLMModelDisplayNames
#from gitbot.main import (
#    agentMessage,
#    streamlit_agent_create_endpoint,
#    streamlit_agent_update_endpoint,
#)
from utils1 import convert_str_to_list, get_indexed_agents
from models import EmbeddingModelDisplayNames, LLMModelDisplayNames
from main import (
    agentMessage,
    streamlit_agent_create_endpoint,
    streamlit_agent_update_endpoint,
)
import time


def layout(
    current_agent_name=None,
    current_repos=None,
    current_include_branches=None,
    current_include_file_types=None,
    current_include_folders=None,
    current_exclude_folders=None,
    current_documentation_folder_path=None,
    current_exclude_file_types=None,
    current_embedding_model_name=None,
    current_llm_model_name=None,
    current_prompt=None,
):
    # 1. Fetch the agent name
    if current_agent_name is not None:
        agent_name = st.text_input(
            "Agent Name", current_agent_name, disabled=True
        )
        is_create = False
    else:
        agent_name = st.text_input(
            "Agent Name", current_agent_name, placeholder="Agent X"
        )
        is_create = True

    # 2. Fetch the repo
    repos = st.text_input(
        "Link to Git Repo",
        current_repos,
        placeholder="https://github.com/org/repo [separated by comma]",
    )
    repos_list = convert_str_to_list(repos)

    # 3. Fetch the branches which needs to be included
    include_branches = st.text_input(
        "Branches to include",
        current_include_branches,
        placeholder="main [separate by comma]",
    )
    include_branches_list = convert_str_to_list(include_branches)

    # 4. Fetch the file types which needs be included
    st.write("File Types to include")

    lst_filetype = list()
    filetypes = [".md", ".json", ".py", ".txt", ".rst"]
    for filetype in filetypes:
        if current_include_file_types is not None:
            if filetype in current_include_file_types:
                if st.checkbox(filetype, value=True):
                    lst_filetype.append(filetype)
            else:
                if st.checkbox(filetype):
                    lst_filetype.append(filetype)
        else:
            if st.checkbox(filetype):
                lst_filetype.append(filetype)
    include_file_types = list(set(lst_filetype))
    include_file_types.sort()

    with st.expander("Click for Advance Options"):

        # 5. Fetch the folders which needs to be included
        include_folders = st.text_input(
            "Folders to include",
            current_include_folders,
            placeholder="app, docs [separate by comma]",
        )
        include_folders_list = convert_str_to_list(include_folders)

        # 6. Fetch the folders which needs to be excluded
        exclude_folders = st.text_input(
            "Folders to exclude",
            current_exclude_folders,
            placeholder="unit_test, notebooks [separate by comma]",
        )
        exclude_folders_list = convert_str_to_list(exclude_folders)
        
        # 7. Fetch the documentation folder path
        documentation_folder_path = st.text_input(
            "Folder path to documentation files if present. Used to generate links to website directly",
            current_documentation_folder_path,
            placeholder="docs",
        )

        # 8. Fetch the folders which needs to be excluded
        exclude_file_types = st.text_input(
            "File types to exclude",
            current_exclude_file_types,
            placeholder=".ipynb, .xml [separate by comma]",
        )
        exclude_file_types = convert_str_to_list(exclude_file_types)

        if current_embedding_model_name is not None:
            model_names = [model.value for model in EmbeddingModelDisplayNames]
            for embedding_model in EmbeddingModelDisplayNames:
                if embedding_model.value == current_embedding_model_name:
                    index = model_names.index(embedding_model.value)
                    break
            embedding_model_name = st.selectbox(
                "Embedding Model", model_names, index=index
            )
        else:
            embedding_model_name = st.selectbox(
                "Embedding Model",
                [model.value for model in EmbeddingModelDisplayNames],
            )

        if current_llm_model_name is not None:
            model_names = [model.value for model in LLMModelDisplayNames]
            for embedding_model in LLMModelDisplayNames:
                if embedding_model.value == current_llm_model_name:
                    index = model_names.index(embedding_model.value)
                    break
            llm_model_name = st.selectbox("LLM Model", model_names, index=index)
        else:
            llm_model_name = st.selectbox(
                "LLM Model", [model.value for model in LLMModelDisplayNames]
            )

        # 9. Custom prompt
        prompt = st.text_area(
            "Custom prompt",
            current_prompt,
            height=300,
            placeholder="Custom Prompt. Be sure to include '{context}' and '{question}' in the prompt",
        )
        if prompt is None:
            prompt = "{query}"

    pressed = st.button("Save")
    if pressed:
        if (
            repos_list is None
            or include_branches_list is None
            or len(include_file_types) == 0
        ):
            st.error("Please fill in the required fields")
        else:
            try:
                with st.spinner("Working on your agent..."):
                    dict_agent = {
                        "agent_name": agent_name,
                        "github_repos": repos_list,
                        "include_branches": include_branches_list,
                        "include_file_types": include_file_types,
                        "exclude_file_types": exclude_file_types,
                        "include_folders": include_folders_list,
                        "exclude_folders": exclude_folders_list,
                        "documentation_folder_path": documentation_folder_path,
                        "embedding_model_name": embedding_model_name,
                        "llm_model_name": llm_model_name,
                        "prompt": prompt,
                    }
                    if is_create:
                        streamlit_agent_create_endpoint(
                            agentMessage(**dict_agent)
                        )
                        st.write("agent created!")
                        agents = get_indexed_agents()
                        agents.sort()
                        st.session_state["agents"] = agents
                    else:
                        is_llm_only_update = True
                        if (
                            convert_str_to_list(current_repos) != repos_list
                            or convert_str_to_list(current_include_branches)
                            != include_branches_list
                            or current_include_file_types != include_file_types
                            or convert_str_to_list(current_include_folders)
                            != include_folders_list
                            or convert_str_to_list(current_exclude_folders)
                            != exclude_folders_list
                            or current_documentation_folder_path
                            != documentation_folder_path
                            or convert_str_to_list(current_exclude_file_types)
                            != exclude_file_types
                            or current_embedding_model_name != embedding_model_name
                        ):
                            is_llm_only_update = False
                        streamlit_agent_update_endpoint(
                            agentMessage(**dict_agent), is_llm_only_update
                        )
                        st.write("agent updated!")

                st.balloons()
                time.sleep(3)
                if not is_create:
                    st.rerun()
            except Exception as e:
                st.error(
                    f"{e} - Please check the details provided!"
                )
