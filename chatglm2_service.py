import os
import json
from flask import Flask
from flask import request
from model import ChatGLMModel

chatglm2 = ChatGLMModel("chatglm2-6b")

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def root():
    """root
    """
    return "Welcome to chatglm model."


@app.route("/chat", methods=["POST"])
def chat():
    """chat
    """
    data_seq = request.get_data()
    data_dict = json.loads(data_seq)
    human_input = data_dict["human_input"]
    history = data_dict['history']
    response, history = chatglm2(human_input, history=history)

    result_dict = {
        "response": response
    }
    result_seq = json.dumps(result_dict, ensure_ascii=False)
    return result_seq


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8595, debug=False)
