from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv


from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler

from queue import Queue

from threading import Thread


load_dotenv()

# queue = Queue()

class StreamingHandler(BaseCallbackHandler):

    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        # print(token)
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, response, **kwargs):
        self.queue.put(None)



chat = ChatOpenAI(
    streaming = True, # streaming = True forces OpenAI to langchain data streanm
    # callbacks = [StreamingHandler()]
    ) 

prompt = ChatPromptTemplate.from_messages([
    ("human", "{content}" )
]
)


# messages = prompt.format_messages(content = "tell me a joke")

# # output = chat(messages) # same as chat.__call__(messages)
# # print(output)

# # output = chat.stream(messages) # stream function forces data from OpenAI to langchain data streanm (overrides streaming = True i, e even streaming = False will stream) and langchain to user stream

# for messages in chat.stream(messages):
#     print(messages)

#     print(messages.content)



# chain = LLMChain(
#     llm = chat,
#     prompt = prompt
# )

# # output = chain("tell me a joke")
# # print(output)


# for output in chain.stream(input = {"content": "tell me a joke"}):
#     print(output) # just get back the complete generator object only, to get streaming valuesb we should override the chain.stream function



class StreamingChain(LLMChain):
    def stream(self, input):
        # self(input) # run the chain

        queue = Queue()
        handler = StreamingHandler(queue)
        def task():
            self(input, callbacks = [handler] ) # run the chain, passing handler can be done here also, different instance will be passesd for each call

        Thread(target = task).start()

        while True:
            token = queue.get()
            if token == None:
                break
            yield token

chain = StreamingChain(
    llm = chat, prompt = prompt
)


# Another way of creating a StreamingChain

class StreamableChain():
    def stream(self, input):

        queue = Queue()
        handler = StreamingHandler(queue)
        def task():
            self(input, callbacks = [handler] ) 
        Thread(target = task).start()

        while True:
            token = queue.get()
            if token == None:
                break
            yield token

class StreamingChain(StreamableChain, LLMChain):
    pass
    
# class StreamingConversationalRetrievalChain(StreamableChain, LLMCConversationalRetrievalChainhain): # eg for using any other type of chains
#     pass
    

chain = StreamingChain(
    llm = chat, prompt = prompt
)

for output in chain.stream(input = {"content": "tell me a joke"}):
    print(output)