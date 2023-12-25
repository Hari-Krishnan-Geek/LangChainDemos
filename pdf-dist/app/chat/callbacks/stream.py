from langchain.callbacks.base import BaseCallbackHandler


class StreamingHandler(BaseCallbackHandler):
    

    def __init__(self, queue):
        self.queue = queue
        self.streaming_run_ids = []

    def on_llm_new_token(self, token, **kwargs):
        # print(token)
        self.queue.put(token)

    def on_llm_end(self, response, run_id,  **kwargs):
        if run_id in self.streaming_run_ids:
            self.queue.put(None)
            self.streaming_run_ids.remove(run_id)



    def on_llm_error(self, response, **kwargs):
        self.queue.put(None)

    def on_chat_model_start(self, serialized, essage, run_id, **kwargs):
        # print(serialized)
        # print(run_id)
        if serialized["kwargs"]["streaming"]:
            self.streaming_run_ids.add(run_id)