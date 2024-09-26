import sys
from pathlib import Path
from icecream import ic
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms.function_calling import FunctionCallingLLM

from src.settings import Settings as ContextualRagSettings

load_dotenv()


class ContextualRagReactAgent:
    def __init__(
        self,
        setting: ContextualRagSettings,
        tools: list[FunctionTool],
        llm: FunctionCallingLLM | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:

        self.setting = setting

        self.llm = llm or self.load_model(self.setting.service, self.setting.model)
        Settings.llm = self.llm

        self.tools = tools

        self.verbose = verbose

        self.query_engine = self.create_query_engine()

        ic("ContextualRagReactAgent initialized !!!")

    @classmethod
    def from_tools(
        cls,
        setting: ContextualRagSettings,
        tools: list[FunctionTool],
        llm: FunctionCallingLLM | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        return cls(
            setting=setting,
            tools=tools,
            llm=llm,
            verbose=verbose,
            **kwargs,
        )

    def load_model(self, service: str, model: str) -> FunctionCallingLLM:
        if service == "openai":
            return OpenAI(model=model)
        else:
            raise ValueError(f"Service {service} not supported.")

    def add_tool(self, tool: FunctionTool) -> None:
        assert isinstance(tool, FunctionTool)

        self.tools.append(tool)

    def create_query_engine(self) -> ReActAgent:
        return ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=self.verbose,
        )

    def predict(self, query: str) -> str:
        return self.query_engine.chat(query)

    async def apredict(self, query: str) -> str:
        return await self.query_engine.achat(query)
