import requests
from dotenv import load_dotenv
import os
load_dotenv()

submission_path = "submission.csv"

ngrok_id = os.getenv("NGROK_ID")
team_name = os.getenv("TEAM_NAME")

server_url = f"https://{ngrok_id}-145-94-238-64.ngrok-free.app/submit"

with open(submission_path, "rb") as f:
    files = {"file": (submission_path, f)}
    data = {"team_name": team_name}
    response = requests.post(server_url, files=files, data=data)

if response.ok:
    print("Server response:", response.json())
else:
    print("Failed to submit:", response.text)
