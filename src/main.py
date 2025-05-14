import boto3
import pickle
import os
import json

from environs import Env
#from gitbot.chatbot import invoke_chatbot
#from gitbot.vector_builder import build_vector
#from gitbot.github_extractor import build_txt_files

from chatbot import invoke_chatbot
from vector_builder import build_vector
from github_extractor import build_txt_files

from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel
from typing import Union, Optional

#from gitbot.models import (
from models import (
    LLMModelDisplayNames,
    LLMModel,
    EmbeddingModel,
    EmbeddingModelDisplayNames,
)

#from gitbot.utils import (
from utils1 import (
    get_indexed_agents, 
    is_agent_indexed, 
    get_model_id, 
    save_indexed_agents,
    save_config
)

from dotenv import load_dotenv
load_dotenv()


class agentMessage(BaseModel):
    agent_name: str
    github_repos: Union[str, list]
    include_branches: Union[str, list]
    include_file_types: Optional[Union[str, list]] = None
    exclude_file_types: Optional[Union[str, list]] = None
    include_folders: Optional[Union[str, list]] = None
    exclude_folders: Optional[Union[str, list]] = None
    documentation_folder_path: Optional[str] = None
    embedding_model_name: Optional[EmbeddingModelDisplayNames] = (
        EmbeddingModelDisplayNames.AMAZON_TITAN_V2.value
    )
    llm_model_name: Optional[LLMModelDisplayNames] = (
        LLMModelDisplayNames.CLAUDE_SONNET_3.value
    )
    prompt: Optional[str] = ("{query}",)
    is_create: Optional[bool] = True


class ChatbotMessage(BaseModel):
    agent_name: str
    query: str


# FastAPI server below #
tags_metadata = [
    {"name": "/healthcheck", "description": "Health check for the API"},
    {
        "name": "/v0",
        "description": "Endpoint that returns a prediction based on the input data",
    },
]

app = FastAPI(
    title="Gitbot",
    description="This is a tool that traverses through provided GitHub repositories, builds agents and have a chatbot respond to questions.",
    openapi_tags=tags_metadata,
)


@app.on_event("startup")
def startup_event():
    print("Starting up the server...")


@app.get("/healthcheck", tags=["/healthcheck"])
async def healthcheck():
    return {"healthy": "true"}


@app.post("/create_agent", tags=["/create_agent"])
async def api_agent_create_endpoint(incoming_data: agentMessage):
    agent_name = jsonable_encoder(incoming_data.agent_name)
    github_repos = jsonable_encoder(incoming_data.github_repos)
    include_branches = jsonable_encoder(incoming_data.include_branches)
    include_folders = jsonable_encoder(incoming_data.include_folders)
    exclude_folders = jsonable_encoder(incoming_data.exclude_folders)
    documentation_folder_path = jsonable_encoder(incoming_data.documentation_folder_path)
    include_file_types = jsonable_encoder(incoming_data.include_file_types)
    exclude_file_types = jsonable_encoder(incoming_data.exclude_file_types)
    prompt = jsonable_encoder(incoming_data.prompt)
    embedding_model_name = jsonable_encoder(incoming_data.embedding_model_name)
    llm_model_name = jsonable_encoder(incoming_data.llm_model_name)
    return agent_creation(
            agent_name,
            github_repos,
            include_branches,
            include_folders,
            exclude_folders,
            documentation_folder_path,
            include_file_types,
            exclude_file_types,
            embedding_model_name,
            llm_model_name,
            prompt,
        )
    
@app.post("/edit_agent", tags=["/edit_agent"])
async def api_agent_edit_endpoint(incoming_data: agentMessage):
    agent_name = jsonable_encoder(incoming_data.agent_name)
    github_repos = jsonable_encoder(incoming_data.github_repos)
    include_branches = jsonable_encoder(incoming_data.include_branches)
    include_folders = jsonable_encoder(incoming_data.include_folders)
    exclude_folders = jsonable_encoder(incoming_data.exclude_folders)
    documentation_folder_path = jsonable_encoder(incoming_data.documentation_folder_path)
    include_file_types = jsonable_encoder(incoming_data.include_file_types)
    exclude_file_types = jsonable_encoder(incoming_data.exclude_file_types)
    prompt = jsonable_encoder(incoming_data.prompt)
    embedding_model_name = jsonable_encoder(incoming_data.embedding_model_name)
    llm_model_name = jsonable_encoder(incoming_data.llm_model_name)
    return agent_update(
            agent_name,
            github_repos,
            include_branches,
            include_folders,
            exclude_folders,
            documentation_folder_path,
            include_file_types,
            exclude_file_types,
            embedding_model_name,
            llm_model_name,
            prompt,
        )


def streamlit_agent_create_endpoint(incoming_data: agentMessage):
    agent_name = incoming_data.agent_name
    github_repos = incoming_data.github_repos
    include_branches = incoming_data.include_branches
    include_folders = incoming_data.include_folders
    exclude_folders = incoming_data.exclude_folders
    documentation_folder_path = incoming_data.documentation_folder_path
    include_file_types = incoming_data.include_file_types
    exclude_file_types = incoming_data.exclude_file_types
    prompt = incoming_data.prompt
    embedding_model_name = incoming_data.embedding_model_name
    llm_model_name = incoming_data.llm_model_name

    return agent_creation(
        agent_name,
        github_repos,
        include_branches,
        include_folders,
        exclude_folders,
        documentation_folder_path,
        include_file_types,
        exclude_file_types,
        embedding_model_name,
        llm_model_name,
        prompt,
    )
        
def streamlit_agent_update_endpoint(incoming_data: agentMessage, is_llm_only_update: bool):
    agent_name = incoming_data.agent_name
    github_repos = incoming_data.github_repos
    include_branches = incoming_data.include_branches
    include_folders = incoming_data.include_folders
    exclude_folders = incoming_data.exclude_folders
    documentation_folder_path = incoming_data.documentation_folder_path
    include_file_types = incoming_data.include_file_types
    exclude_file_types = incoming_data.exclude_file_types
    prompt = incoming_data.prompt
    embedding_model_name = incoming_data.embedding_model_name
    llm_model_name = incoming_data.llm_model_name
    is_llm_only_update = is_llm_only_update
    return agent_update(
        agent_name,
        github_repos,
        include_branches,
        include_folders,
        exclude_folders,
        documentation_folder_path,
        include_file_types,
        exclude_file_types,
        embedding_model_name,
        llm_model_name,
        prompt,
        is_llm_only_update,
    )


def build_config(
    github_repos,
    include_branches,
    include_folders,
    exclude_folders,
    documentation_folder_path,
    include_file_types,
    exclude_file_types,
    embedding_model_name,
    llm_model_name,
    embedding_model_id,
    llm_model_id,
    prompt,
):
    return {
        "github_repos": github_repos,
        "include_branches": include_branches,
        "include_folders": include_folders,
        "exclude_folders": exclude_folders,
        "documentation_folder_path": documentation_folder_path,
        "include_file_types": include_file_types,
        "exclude_file_types": exclude_file_types,
        "embedding_model_name": embedding_model_name.value,
        "llm_model_name": llm_model_name.value,
        "embedding_model_id": embedding_model_id,
        "llm_model_id": llm_model_id,
        "prompt": prompt,
    }


def index_agent(agents, agent_name):
    agents.append(agent_name)
    save_indexed_agents(agents)


def agent_creation(
    agent_name,
    github_repos,
    include_branches,
    include_folders,
    exclude_folders,
    documentation_folder_path,
    include_file_types,
    exclude_file_types,
    embedding_model_name,
    llm_model_name,
    prompt,
):
    env = Env()
    env.read_env()  # this loads from .env
    auth_token = env.str("GITHUB_AUTH_TOKEN")
    print("Loaded GitHub token:", bool(auth_token))

    try:
        agents = get_indexed_agents()
    except:
        print("Good for now")
        agents = []

    if is_agent_indexed(agents, agent_name):
        raise ValueError(f"agent {agent_name} already exists.")
    
    print("Before embedding_model_id")
    embedding_model_id = get_model_id(
        embedding_model_name, EmbeddingModelDisplayNames, EmbeddingModel
    )
    llm_model_id = get_model_id(llm_model_name, LLMModelDisplayNames, LLMModel)
    print("Emb model id: ", embedding_model_id)
    print("LLM model name: ", llm_model_name)
    print("LLM Model id: ", llm_model_id)
    print("Prompt: ", prompt)

    print("Before build_config")
    config = build_config(
        github_repos,
        include_branches,
        include_folders,
        exclude_folders,
        documentation_folder_path,
        include_file_types,
        exclude_file_types,
        embedding_model_name,
        llm_model_name,
        embedding_model_id,
        llm_model_id,
        prompt,
    )
    print("Before build_txt_files")
    #build_txt_files(
    #    github_repos,
    #    auth_token,
    #    include_branches,
    #    include_folders,
    #    exclude_folders,
    #    documentation_folder_path,
    #    include_file_types,
    #    exclude_file_types,
    #)
    
    build_txt_files(
        repos=github_repos,
        include_branches=include_branches,
        include_folders=include_folders,
        exclude_folders=exclude_folders,
        documentation_folder_path=documentation_folder_path,
        include_file_types=include_file_types,
        exclude_file_types=exclude_file_types,
        auth_token=auth_token,
    )
    
    print("Before build_vector")
    #print("Agent name: ", agent_name)
    print("Emb model name: ", embedding_model_id)
    print("LLM model name: ", llm_model_name)
    print("Prompt: ", prompt)
    print("Auth token: ", auth_token)
    build_vector(agent_name, embedding_model_id, llm_model_name, prompt)
    #build_vector(agent_name, embedding_model_id, llm_model_name, prompt, auth_token)

    save_config(agent_name, config)

    index_agent(agents, agent_name)

    return {"status": "agent created successfully."}


def agent_update(
    agent_name,
    github_repos,
    include_branches,
    include_folders,
    exclude_folders,
    documentation_folder_path,
    include_file_types,
    exclude_file_types,
    embedding_model_name,
    llm_model_name,
    prompt,
    is_llm_only_update
):
    env = Env()
    auth_token = env.str("GITHUB_AUTH_TOKEN")

    agents = get_indexed_agents()

    if not is_agent_indexed(agents, agent_name):
        raise ValueError(f"agent {agent_name} does not exist.")

    embedding_model_id = get_model_id(
        embedding_model_name, EmbeddingModelDisplayNames, EmbeddingModel
    )
    llm_model_id = get_model_id(llm_model_name, LLMModelDisplayNames, LLMModel)

    config = build_config(
        github_repos,
        include_branches,
        include_folders,
        exclude_folders,
        documentation_folder_path,
        include_file_types,
        exclude_file_types,
        embedding_model_name,
        llm_model_name,
        embedding_model_id,
        llm_model_id,
        prompt,
    )

    if not is_llm_only_update:
        build_txt_files(
            github_repos,
            auth_token,
            include_branches,
            include_folders,
            exclude_folders,
            documentation_folder_path,
            include_file_types,
            exclude_file_types,
        )
        build_vector(agent_name, 
                    embedding_model_id, llm_model_name, 
                    prompt)

    save_config(agent_name, config)

    return {"status": "agent updated successfully."}


@app.post("/chatbot", tags=["/chatbot"])
async def chatbot(incoming_data: ChatbotMessage):
    agent_name = jsonable_encoder(incoming_data.agent_name)
    query = jsonable_encoder(incoming_data.query)

    invoke_chatbot(agent_name, query)

    return invoke_chatbot(agent_name, query)
