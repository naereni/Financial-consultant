# нижеследующие преобразования необходимы для передачи истории сообщений
# в сценарий CMS, допустимы только простые типы


# преобразование списков объектов в простые типы
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def list_to_dict(instance):
    if not instance:
        return []

    output = []
    for item in instance:
        if isinstance(item, list):
            output.append(list_to_dict(item))
        else:
            output.append(serialize(item))
    return output


# преобразование объектов в простые типы
def serialize(instance):
    if instance is None:
        return {}

    output = {}
    if not isinstance(instance, dict):
        items = instance.__dict__.items()
    else:
        items = instance.items()
    for key, value in items:
        if isinstance(value, BaseChatMessageHistory) or isinstance(value, BaseMessage):
            output["classname"] = type(value).__name__
            output[key] = serialize(value.__dict__)
        elif isinstance(value, list):
            output[key] = list_to_dict(value)
        else:
            if isinstance(instance, BaseMessage):
                output["classname"] = type(instance).__name__
            output[key] = value
    return output


# преобразование в обьекты классов
def deserialize(dictionary):
    if not dictionary:
        return None

    human_prefix: str = "human"
    ai_prefix: str = "ai"

    dict_input_key = dictionary["input_key"]
    dict_memory_key = dictionary["memory_key"]

    cbm = ConversationBufferMemory(memory_key=dict_memory_key, input_key=dict_input_key)
    cbm.chat_memory = ChatMessageHistory()

    for item in dictionary["chat_memory"]["messages"]:
        if item["type"] == human_prefix:
            hm = HumanMessage(item["content"])
            cbm.chat_memory.add_message(hm)
        elif item["type"] == ai_prefix:
            aim = AIMessage(item["content"])
            cbm.chat_memory.add_message(aim)
    return cbm
