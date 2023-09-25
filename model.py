from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import LLM
from pydantic import Field
from typing import Any

logger = logging.get_logger(__name__)


class ChatGLMModel(LLM):
    path_or_name: str = Field(None, alias='path_or_name')
    tokenizer: Any = None
    model: Any = None

    def __init__(self, path_or_name):
        super(ChatGLMModel, self).__init__()
        self.path_or_name = path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path_or_name, trust_remote_code=True).half().cuda()
        self.model.eval()

    @property
    def _llm_type(self):
        return "custom"

    @property
    def _identifying_params(self):
        return {
            "path_or_name": self.path_or_name
        }

    def _call(self, prompt, stop):
        response, _ = self.model.chat(self.tokenizer, prompt)
        return response


if __name__ == "__main__":
    chatglm2 = ChatGLMModel("chatglm2-6b")

    template = """你是一个和Human对话的机器人AI。根据历史的对话情况回答Human的问题。
    {chat_history}

    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    memory.chat_memory.add_user_message("你好!")
    memory.chat_memory.add_ai_message("你好， 我是你的私人AI助手小I")

    memory.chat_memory.add_user_message("我叫张骞，你叫什么？")
    memory.chat_memory.add_ai_message("我叫小I")

    llm_chain = LLMChain(
        llm=chatglm2,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    human_input = "我叫什么？你叫什么？"
    response = llm_chain.predict(human_input=human_input)
    print(response)
