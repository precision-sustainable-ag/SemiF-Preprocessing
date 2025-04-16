# scripts/create_failure_issue.py
import os
import requests

token = os.environ['GH_TOKEN']
repo = os.environ['REPO']
batch_id = os.environ['BATCH_ID']
assignee = os.environ['ASSIGNEE']
globus_prefix = os.environ['GLOBUS_PREFIX']
task_name = os.environ.get('TASK_NAME', 'Unknown')
error_msg = os.environ.get('ERROR_MSG', 'No message provided')

globus_log = f"{globus_prefix}/{batch_id}/inspection/{batch_id}.log"

title = f"{batch_id}: Task Failed - {task_name}"
body = (
    f"### Task `{task_name}` failed for batch `{batch_id}`\n\n"
    f"**Error message:**\n```\n{error_msg}\n```\n\n"
    f"[Click here to view the log]({globus_log})\n\n"
    f"_Assigned to @{assignee}_"
)

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

payload = {
    "title": title,
    "body": body,
    "assignees": [assignee]
}

response = requests.post(
    f"https://api.github.com/repos/{repo}/issues",
    headers=headers,
    json=payload
)

if response.ok:
    print("Failure issue created:", response.json()['html_url'])
else:
    print("Failed to create issue:", response.status_code, response.text)
    exit(1)
