# scripts/create_issue.py

import os
import requests

token = os.environ['GH_TOKEN']
repo = os.environ['REPO']
batch_id = os.environ['BATCH_ID']
assignee = os.environ['ASSIGNEE']
globus_prefix = os.environ['GLOBUS_PREFIX']
print(f"Token length: {len(token)}")
print(f"Token starts with: {token[:4]}")
globus_pdf_report = f"{globus_prefix}/{batch_id}/inspection/{batch_id}_report.pdf"
globus_log = f"{globus_prefix}/{batch_id}/inspection/{batch_id}.log"

title = f"{batch_id}: Inspection Report"
body = (
    "### New Report Available\n\n"
    f"The inspection report for batch `{batch_id}` is now available.\n\n"
    f"[Click here to view the report]({globus_pdf_report})\n\n"
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
    print("✅ Issue created:", response.json()['html_url'])
else:
    print("❌ Failed:", response.status_code, response.text)
    exit(1)
