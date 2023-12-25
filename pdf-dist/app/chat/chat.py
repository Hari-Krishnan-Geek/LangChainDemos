from app.chat.models import ChatArgs
# from app.chat.vector_stores.pinecone import build_retriever
from app.chat.vector_stores import retriever_map
from langchain.chains import ConversationalRetrievalChain
# from app.chat.llms.chatopenai import build_llm
from app.chat.llms import llm_map
# from app.chat.memories.sql_memory import build_memory
from app.chat.memories import memory_map

from app.chat.chains.retrieval import StreamingConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI

from app.web.api import (
    set_conversation_components,
    get_conversation_components
)
import random
from app.chat.score import random_component_by_choice




def select_component(component_type, component_map, chat_args):
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]

    if previous_component:
            # not first retriever, so I need to use the same previous one again
            builder = component_map[previous_component]
            return previous_component, builder(chat_args)

    else:
        # first message, so taking random one
        # random_name = random.choice(list(component_map.keys()))

        random_name = random_component_by_choice(component_type, component_map)
        builder = component_map[random_name]

        return random_name, builder(chat_args)

        # retriver = build_retriever(chat_args)
        # set_conversation_components(
        #     conversation_id = chat_args.conversation_id,
        #     llm = "",
        #     memory = "",
        #     retriver = random_retriever_name
        # )


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """

    # retriever = build_retriever(chat_args)


    retriever_name, retriever = select_component(
         "retriever",
         retriever_map,
         chat_args
    )

    llm_name, llm = select_component(
         "llm",
         llm_map,
         chat_args
    )

    memory_name, memory = select_component(
         "memory",
         memory_map,
         chat_args
    )

    # print(f"running with ret: {retriever_name}, llm : {llm_name}, memory : {memory_name}")

    set_conversation_components(
         chat_args.conversation_id,
         llm = llm_name,
         retriever=retriever_name,
         memory=memory_name
    )

    # llm = build_llm(chat_args)
    condense_question_llm = ChatOpenAI(streaming = False)
    # memory = build_memory(chat_args)


    # return ConversationalRetrievalChain.from_llm(
    #     llm = llm,
    #     retriever = retriever,
    #     memory = memory

    # )


    return StreamingConversationalRetrievalChain.from_llm(
        llm = llm,
        condense_question_llm = condense_question_llm,
        retriever = retriever,
        memory = memory
    )