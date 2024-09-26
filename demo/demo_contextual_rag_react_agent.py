import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.service import ContextualRagReactAgent

parser = argparse.ArgumentParser(description="Demo Contextual Rag React Agent")
parser.add_argument("--q", type=str, help="Query to ask the agent", required=True)
args = parser.parse_args()

agent = ContextualRagReactAgent()

print(agent.complete(args.q))
