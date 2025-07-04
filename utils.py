from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from py2neo import Graph
from config import *

import os
from dotenv import load_dotenv
load_dotenv()

def get_embeddings_model():
    model_map = {
        'openai':OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDINGS_MODEL')
        )
    }
    return model_map[os.getenv('EMBEDDINGS_MODEL')]

def get_llm_model():
    model_map = {
        'openai':ChatOpenAI(
            model=os.getenv('OPENAI_LLM_MODEL')
        )
    }
    return model_map[os.getenv('LLM_MODEL')]

def structured_output_parser(response_schemas):
    text =  '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段。
    注意：必须包含下列所有字段！！字段没有实体信息填充为空列表！！：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text

def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%',value)
    return string

def get_neo4j_conn():
    return Graph(
        os.getenv('NEO4J_URI'),
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )