import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.react_agent import ContextualRagReactAgent
from src.tools.contextual_rag_tool import load_contextual_rag_tool

parser = argparse.ArgumentParser(description="Demo Contextual Rag React Agent")
parser.add_argument("--q", type=str, help="Query to ask the agent")
args = parser.parse_args()

agent = ContextualRagReactAgent.from_tools(
    tools=[load_contextual_rag_tool()],
    verbose=True,
)

print(agent.predict(args.q))
