import requests

submission_path = "submission.csv"

ngrok_id = "..."
server_url = f"https://{ngrok_id}-145-94-238-64.ngrok-free.app/submit"

with open(submission_path, "rb") as f:
    files = {"file": (submission_path, f)}
    response = requests.post(server_url, files=files)

if response.ok:
    print("Server response:", response.json())
else:
    print("Failed to submit:", response.text)
