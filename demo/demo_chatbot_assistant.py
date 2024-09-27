import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.service import ChatbotAssistant

parser = argparse.ArgumentParser(description="Demo Chat Bot Assistant")
parser.add_argument("--q", type=str, help="Query to ask chat bot", required=True)
args = parser.parse_args()

bot = ChatbotAssistant()

print(bot.complete(args.q))
