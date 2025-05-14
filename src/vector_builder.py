import boto3
import os
import json
import pickle

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from pydantic import validate_call

import sys


@validate_call
def build_vector(
    agent_name: str,
    embedding_model_id: str,
    llm_model_name: str,
    prompt: str = None,
    files_src_dir: str = "./tmp/",
):

    loader = DirectoryLoader(files_src_dir, show_progress=True, loader_cls=TextLoader)
    repo_files = loader.load()
    print(f"Number of files loaded: {len(repo_files)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    documents = text_splitter.split_documents(documents=repo_files)
    print(f"Number of documents : {len(documents)}")

    for doc in documents:
        old_path_with_txt_extension = doc.metadata["source"]
        new_path_without_txt_extension = old_path_with_txt_extension.replace(".txt", "")
        doc.metadata.update({"source": new_path_without_txt_extension})

    bc = boto3.client("bedrock-runtime", region_name="us-east-1")

    bedrock_embeddings = BedrockEmbeddings(model_id=embedding_model_id, client=bc)

    vectorstore_faiss = FAISS.from_documents(
        documents,
        bedrock_embeddings,
    )

    vectorstore_faiss.save_local(agent_name)

    s3 = boto3.client("s3")
    s3.upload_file(
        Filename="./" + agent_name + "/index.faiss",
        #Bucket="sagemaker-us-east-2-534295958235",
        Bucket="myprojects-2025",
        Key=f"gitbot/{agent_name}/index.faiss",
    )
    s3.upload_file(
        Filename="./" + agent_name + "/index.pkl",
        #Bucket="sagemaker-us-east-2-534295958235",
        Bucket="myprojects-2025",
        Key=f"gitbot/{agent_name}/index.pkl",
    )

    # Temporarily deactivated
    if prompt is None:
        prompt = """You are a RAG based virtual assistant named GitBot. Users will ask questions about a particular Git repo whose content is available to you in the vector embedding. 
        You cannot use information outside of the knowledge base to answer the question. 
        Only give a concise answer. 
        User asked this question: {query}.
        """
