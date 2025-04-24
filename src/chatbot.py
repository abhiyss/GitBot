from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains.question_answering import load_qa_chain
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import FAISS
import os

import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import PyPDFLoader
import boto3
import sys


def load_embedding(bc, agent_name: str):
    if not os.path.exists(agent_name + "-chatbot"):
        os.makedirs(agent_name + "-chatbot", exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(
        Bucket="sagemaker-us-east-2-534295958235",
        Key=f"gitbot/{agent_name}/index.faiss",
        Filename="./" + agent_name + "-chatbot/index.faiss",
    )
    s3.download_file(
        Bucket="sagemaker-us-east-2-534295958235",
        Key=f"gitbot/{agent_name}/index.pkl",
        Filename="./" + agent_name + "-chatbot/index.pkl",
    )
    config = (
        s3.get_object(
            Bucket="sagemaker-us-east-2-534295958235",
            Key=f"gitbot/{agent_name}/config.json",
        )["Body"]
        .read()
        .decode("utf-8")
    )
    config = json.loads(config)
    embedding_model_id, llm_model_id, prompt = (
        config["embedding_model_id"],
        config["llm_model_id"],
        config["prompt"],
    )
    bedrock_embeddings = BedrockEmbeddings(model_id=embedding_model_id, client=bc)

    vectorstore_faiss = FAISS.load_local(
        agent_name + "-chatbot",
        bedrock_embeddings,
        allow_dangerous_deserialization=True,
    )

    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

    return vectorstore_faiss, wrapper_store_faiss, llm_model_id, prompt


def set_llm(bc, model_id):
    llm = ChatBedrock(
        model_id=model_id, client=bc
    )  # , model_kwargs=dict(temperature=0))
    return llm


def document_data(query, chat_history):

    pdf_path = "<Pdf Path>"
    loader = PyPDFLoader(file_path=pdf_path)
    doc = loader.load()


def invoke_chatbot(agent_name: str, query: str, chat_history=None):
    bc = boto3.client("bedrock-runtime", region_name="us-east-1")

    vectorstore_faiss, wrapper_store_faiss, llm_model_id, prompt = load_embedding(
        bc, agent_name
    )

    llm = set_llm(bc, llm_model_id)

    if prompt != "{query}":
        PROMPT = PromptTemplate(
            template=prompt, input_variables=["context", "question"]
        )
        if chat_history is not None:
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore_faiss.as_retriever(
                    search_type="similarity", search_kwargs={"k": 9}
                ),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT},
            )
            result = qa({"question": query, "chat_history": chat_history})
            return result
        else:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore_faiss.as_retriever(
                    search_type="similarity", search_kwargs={"k": 9}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
            )
            result = qa({"query": query})
            answer = result["result"]

    else:
        final_query = prompt.format(query=query)
        answer = wrapper_store_faiss.query(question=final_query, llm=llm)
    return {"answer": answer}
