import os
import logging
from icecream import ic
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini

from src.agents.react_agent import ReActAgent
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from starlette.responses import StreamingResponse, Response
from src.tools.contextual_rag_tool import load_contextual_rag_tool
from src.constants import (
    SERVICE,
    TEMPERATURE,
    MODEL_ID,
    STREAM,
    AGENT_TYPE,
)

load_dotenv(override=True)


class ChatbotAssistant:
    query_engine: AgentRunner
    tools_dict: dict

    def __init__(self):
        self.tools = self.load_tools()
        self.query_engine = self.create_query_engine()

    def load_tools(self):
        """
        Load default RAG tool.
        """
        contextual_rag_tool = load_contextual_rag_tool()
        return [contextual_rag_tool]

    def add_tools(self, tools: FunctionTool | list[FunctionTool]) -> None:
        """
        Add more tools to the agent.

        Args:
            tools (FunctionTool | list[FunctionTool]): A single tool or a list of tools to add to the agent.
        """
        if isinstance(tools, FunctionTool):
            tools = [tools]

        self.tools.extend(tools)
        ic(f"Add: {len(tools)} tools.")

        self.query_engine = (
            self.create_query_engine()
        )  # Re-create the query engine with the new tools

    def create_query_engine(self):
        """
        Creates and configures a query engine for routing queries to the appropriate tools.

        This method initializes and configures a query engine for routing queries to specialized tools based on the query type.
        It loads a language model, along with specific tools for tasks such as code search and paper search.

        Returns:
            AgentRunner: An instance of AgentRunner configured with the necessary tools and settings.
        """

        llm = self.load_model(SERVICE, MODEL_ID)
        Settings.llm = llm

        ic(AGENT_TYPE, len(self.tools))

        if AGENT_TYPE == "react":
            query_engine = ReActAgent.from_tools(
                tools=self.tools, verbose=True, llm=llm
            )
        elif AGENT_TYPE == "openai":
            query_engine = OpenAIAgent.from_tools(
                tools=self.tools, verbose=True, llm=llm
            )
        else:
            raise ValueError("Unsupported agent type!")

        return query_engine

    def load_model(self, service, model_id):
        """
        Select a model for text generation using multiple services.
        Args:
            service (str): Service name indicating the type of model to load.
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
        Returns:
            LLM: llama-index LLM for text generation
        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        logging.info(f"Loading Model: {model_id}")
        logging.info("This action can take a few minutes!")

        if service == "ollama":
            logging.info(f"Loading Ollama Model: {model_id}")
            return Ollama(model=model_id, temperature=TEMPERATURE)
        elif service == "openai":
            logging.info(f"Loading OpenAI Model: {model_id}")
            return OpenAI(
                model=model_id,
                temperature=TEMPERATURE,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif service == "groq":
            logging.info(f"Loading Groq Model: {model_id}")
            return Groq(
                model=model_id,
                temperature=TEMPERATURE,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        elif service == "gemini":
            logging.info(f"Loading Gemini Model: {model_id}")
            return Gemini(
                model=model_id,
                temperature=TEMPERATURE,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            raise NotImplementedError(
                "The implementation for other types of LLMs are not ready yet!"
            )

    def complete(self, query: str) -> str:
        """
        Generate response for given query.

        Args:
            query (str): The input query.

        Returns:
            str: The response.
        """
        return self.query_engine.chat(query)

    def predict(self, prompt):
        """
        Predicts the next sequence of text given a prompt using the loaded language model.

        Args:
            prompt (str): The input prompt for text generation.

        Returns:
            str: The generated text based on the prompt.
        """
        # Assuming query_engine is already created or accessible
        if STREAM:
            # self.query_engine.memory.reset()
            streaming_response = self.query_engine.stream_chat(prompt)

            return StreamingResponse(
                streaming_response.response_gen,
                media_type="application/text; charset=utf-8",
            )
            # return StreamingResponse(streaming_response.response_gen, media_type="application/text; charset=utf-8")

        else:
            return Response(
                self.query_engine.chat(prompt).response,
                media_type="application/text; charset=utf-8",
            )
