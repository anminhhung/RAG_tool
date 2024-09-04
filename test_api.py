import requests
import json
from datetime import time


# Base URL (replace with your actual API URL)
base_url = "http://localhost:8000/v1"  # Example: localhost

# 1. Test /chat endpoint
complete_url = f"{base_url}/chat"
message_data = {"message": "Hello"}
response = requests.post(complete_url, json=message_data)
print("response", response)
if response.status_code == 200:
    print("'/chat' response:", response.json())
else:
    print(f"Error calling '/chat': {response.status_code} - {response.text}")
