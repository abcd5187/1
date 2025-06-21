from utils import *
from config import *
from prompt import *

import os
from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain.chains.llm_math.base import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

class Agent():
    def __init__(self):
        self.vdb = Chroma(
            embedding_function=get_embeddings_model(),
            persist_directory=os.getenv('DB_PATH')
        )
    def generic_func(self, query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        return chain.invoke(query)['text']

    def retrival_func(self, query):
        # 召回并过滤文档
        documents = self.vdb.similarity_search_with_relevance_scores(query=query,k=5)
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]
        
        # 填充提示词并总结答案
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query':query,
            'query_result':'\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return chain.invoke(inputs)['text']

    def graph_func(self, query):
        # 命名体识别
        response_schemas = [
            ResponseSchema(type='list',name='disease',description='疾病名称实体'),
            ResponseSchema(type='list',name='symptom',description='疾病症状实体'),
            ResponseSchema(type='list',name='drug',description='药品名称实体')
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)
        prompt = PromptTemplate.from_template(NER_PROMPT_TPL)
        chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            output_parser=output_parser,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'format_instructions':format_instructions,
            'query':query
        }
        result = chain.invoke(inputs)['text']
        
        # 命名体识别结果，填充模板
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'],[[slot,value]]),
                    'cypher': replace_token_in_string(template['cypher'],[[slot,value]]),
                    'answer': replace_token_in_string(template['answer'],[[slot,value]])
                })
        if not graph_templates:
            return
        
        # 计算问题相似度，筛选最相关的问题
        graph_documents = [
            Document(page_content=template['question'], metadata=template) for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents,get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_score(query,k=3)
        
        # 执行CQL，拿到结果
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            result = neo4j_conn.run(cypher).data()
            if result and any(value for value in result[0].values()):
                answer_str = replace_token_in_string(answer, list(result[0].items()))
                query_result.append(f"问题:{question}\n答案:{answer_str}")
        
        # 总结答案
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query':query,
            'query_result':query_result if len(query_result) else '没有查到'
        }
        return chain.invoke(inputs)['text']

    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        llm_request_chain = LLMRequestsChain(
            llm_chain = llm_chain,
            requests_key = 'query_result',
        )
        inputs = {
            'query':query,
            'url':'https://www.so.com/s?q='+query.replace('','+')
        }
        return llm_request_chain.invoke(inputs)['output']
    
    def query(self, query):
        tools = [
            Tool.from_function(
                name='generic_func',
                func=self.generic_func,
                description='可以解答通用领域的知识，例如打招呼，问你是谁等问题'
            ),
            Tool.from_function(
                name='retrival_fun',
                func=self.retrival_func,
                description='用于回答寻医问药网相关问题',
            ),
            Tool(
                name='graph_func',
                func=self.graph_func,
                description='用于回答疾病、症状、药物等医疗相关问题',
            ),
            Tool.from_function(
                name='search_fun',
                func=self.search_func,
                description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题',
            )
        ]


        tool = self.parse_tools(tools,query)
        return tool.func(query)
        exit()

        # prefix = """请用中文，尽你所能回答以下问题。您可以使用以下工具："""
        # suffix = '''Begin!
        #     History: {chat_history}
        #     Question: {input}
        #     Thought:{agent_scratchpad}
        # '''
        # agent_prompt = ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=['chat_history', 'input', 'agent_scratchpad']
        # )
        # llm_chain = LLMChain(llm=get_llm_model(),prompt=agent_prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain)


        
        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = '请用中文回答问题！Final Answer 必须尊重 Obversion 的结果，不能改变语义。\n\n' + prompt.template
        agent = create_react_agent(llm=get_llm_model(), tools=tools, prompt=prompt)

        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose = os.getenv('VERBOSE')
        )

        return agent_chain.invoke({'input':query})['output']

    def parse_tools(self, tools, query):
        prompt = PromptTemplate.from_template(PARSE_TOOLS_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )

        tools_description = ''
        for tool in tools:
            tools_description += tool.name+':'+tool.description+'\n'
        result = llm_chain.invoke({'tools_description':tools_description,'query':query})
        for tool in tools:
            if tool.name == result['text']:
                return tool
        return tools[0] # 如果找不到最合适的工具就默认generic_func