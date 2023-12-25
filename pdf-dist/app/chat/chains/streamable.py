from queue import Queue
from threading import Thread
from app.chat.callbacks.stream import StreamingHandler
from flask import current_app

class StreamableChain():
    def stream(self, input):

        queue = Queue()
        handler = StreamingHandler(queue)
        def task(app_context):
            app_context.push() # needs to pass args and this line since we are using flask app in front end, else it is not needed
            self(input, callbacks = [handler] ) 
        Thread(target = task, args = [current_app.app_context()]).start()

        while True:
            token = queue.get()
            if token == None:
                break
            yield token