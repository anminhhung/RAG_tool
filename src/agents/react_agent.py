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
    """
    ContextualRagReactAgent is a class that acts as a wrapper around the ReActAgent class.
    """

    setting: ContextualRagSettings
    llm: FunctionCallingLLM
    tools: list[FunctionTool]
    verbose: bool
    query_engine: ReActAgent

    def __init__(
        self,
        setting: ContextualRagSettings,
        tools: list[FunctionTool],
        llm: FunctionCallingLLM | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the ContextualRagReactAgent class.

        Args:
            setting (ContextualRagSettings): Settings.
            tools (list[FunctionTool]): List of tools to be used by the agent.
            llm (FunctionCallingLLM, optional): LLM model to be used by the agent. Defaults to `None`.
            verbose (bool, optional): Verbosity of the agent. Defaults to `True`.
        """

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
        """
        Create an instance of the ContextualRagReactAgent class from a list of tools.

        Args:
            setting (ContextualRagSettings): Settings.
            tools (list[FunctionTool]): List of tools to be used by the agent.
            llm (FunctionCallingLLM, optional): LLM model to be used by the agent. Defaults to `None`.
            verbose (bool, optional): Verbosity of the agent. Defaults to `True`.

        Returns:
            ContextualRagReactAgent: An instance of the ContextualRagReactAgent class.
        """
        return cls(
            setting=setting,
            tools=tools,
            llm=llm,
            verbose=verbose,
            **kwargs,
        )

    def load_model(self, service: str, model: str) -> FunctionCallingLLM:
        """
        Load LLM model.

        Args:
            service (str): service to use, e.g., `openai`.
            model (str): model name, e.g., `gpt-4o-mini`.

        Returns:
            FunctionCallingLLM: LLM model.
        """
        if service == "openai":
            return OpenAI(model=model)
        else:
            raise ValueError(f"Service {service} not supported.")

    def add_tool(self, tool: FunctionTool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool (FunctionTool): Tool to be added to the agent.
        """
        assert isinstance(tool, FunctionTool)

        self.tools.append(tool)

        # Recreate the query engine with the updated tools
        self.query_engine = self.create_query_engine()

    def create_query_engine(self) -> ReActAgent:
        """
        Create a query engine.

        Returns:
            ReActAgent: ReactAgent as a query engine.
        """
        return ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=self.verbose,
        )

    def predict(self, query: str) -> str:
        """
        Give a response to the query.

        Args:
            query (str): Query to be responded to.

        Returns:
            str: Response to the query.
        """
        return self.query_engine.chat(query)

    async def apredict(self, query: str) -> str:
        """
        Give a response to the query asynchronously.

        Args:
            query (str): Query to be responded to.

        Returns:
            str: Response to the query.
        """
        return await self.query_engine.achat(query)
