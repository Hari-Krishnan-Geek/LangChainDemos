from .chatopenai import build_llm
from functools import partial


llm_map = {
    "gpt-4": partial(build_llm, model_name = "gpt-4"), # model_name will go as an arg to build_llm b y using partial
    "gpt-3.5-turbo":partial(build_llm, model_name = "gpt-3.5-turbo") 

}

# builder = llm_map["gpt-4"]

# builder(chat_args)