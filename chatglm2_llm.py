from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel


def inference(model, tokenizer):
    pass


class Chatglm2LLM(LLM):
    def __init__(self, base_model):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(base_model, trust_remote_code=True).cuda()

    def _call(self, prompt, stop, run_manager: Optional[CallbackManagerForLLMRun] = None, ) -> str:
        self.log('----------' + self._llm_type + '----------> llm._call()')
        self.log(prompt)

        response = inference(self.model, self.tokenizer, prompt)
        return response
