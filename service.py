from prompt import *
from utils import *
from agent import *

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

class Service():
    def __init__(self):
        self.agent = Agent()

    def get_summary_message(self,message,history):
        llm = get_llm_model()
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TPL)
        llm_chain = LLMChain(llm=llm,prompt=prompt,verbose=True)
        chat_history = ''
        for q, a in history[-2:]:
            chat_history += f'问题{q},答案{a}\n'
        return llm_chain.invoke({'query':message, 'chat_history':chat_history})['text']
    
    def answer(self,message,history):
        if history:
            message = self.get_summary_message(message=message,history=history)
        return self.agent.query(message)