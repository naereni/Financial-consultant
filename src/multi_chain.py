"""Use a single chain to route an input to one of multiple llm chains."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Union

from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Учитывая исходный текстовый ввод в языковую модель и историю диалога, выбери наиболее подходящий запрос для \
ввода. Тебе будут даны имена доступных запросов и описание того, для чего лучше всего подходит \
запрос. 

<< ФОРМАТИРОВАНИЕ >>
Верни фрагмент кода в markdown с объектом JSON, отформатированным таким образом:
```json
{{{{
    "destination": string \\ имя используемого запроса или "DEFAULT"
    "next_inputs": string \\ исходный ввод
}}}}
```

ПОМНИ: "destination" ДОЛЖЕН быть одним из имен кандидатов на запрос, указанных ниже, ИЛИ \
он может быть "DEFAULT", если ввод не очень подходит для любого из кандидатов на запрос.
ПОМНИ: "next_inputs" - просто исходный ввод.

<< КАНДИДАТЫ НА ЗАПРОСЫ >>
{destinations}

<< ИСТОРИЯ ДИАЛОГА >>
{{history}}

<< ВВОД >>
{{question}}

<< ВЫВОД (должен включать ```json в начале ответа) >>
"""


class MultiChain(MultiRouteChain, ABC):
    """A multi-route chain that uses an LLM router chain to choose amongst chains."""

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_chains(
        cls,
        llm: BaseLanguageModel,
        chain_infos: List[Dict[str, Union[str, Chain]]],
        default_chain: Optional[Chain] = None,
        **kwargs: Any,
    ) -> MultiChain:
        """Convenience constructor for instantiating from destination prompts."""
        destinations = [f"{p['name']}: {p['description']}" for p in chain_infos]
        destinations_str = "\n\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["question"],
            output_parser=RouterOutputParser(next_inputs_inner_key="question"),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {}
        for c_info in chain_infos:
            name = c_info["name"]
            chain = c_info["chain"]
            destination_chains[name] = chain
        _default_chain = default_chain or ConversationChain(llm=llm, output_key="text")
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )
