import os
import requests

GITHUB_API = "https://api.github.com"
TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["REPO"]
BATCH_ID = os.environ["BATCH_ID"]
ASSIGNEE = os.environ["ASSIGNEE"]
GLOBUS_PREFIX = os.environ["GLOBUS_PREFIX"]

IS_SUCCESS = os.environ.get("IS_SUCCESS", "false").lower() == "true"
TASK_NAME = os.environ.get("TASK_NAME", "Unknown")
ERROR_MSG = os.environ.get("ERROR_MSG", "No message provided")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def find_existing_issue(batch_id):
    query = f"repo:{REPO} in:title {batch_id}"
    url = f"{GITHUB_API}/search/issues?q={query}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    for item in resp.json().get("items", []):
        if batch_id in item["title"]:
            return item["number"]
    return None

def comment_on_issue(issue_number, message):
    url = f"{GITHUB_API}/repos/{REPO}/issues/{issue_number}/comments"
    payload = {"body": message}
    resp = requests.post(url, json=payload, headers=HEADERS)
    resp.raise_for_status()
    print(f"💬 Commented on issue #{issue_number}")

def create_issue(title, body, assignee):
    url = f"{GITHUB_API}/repos/{REPO}/issues"
    payload = {
        "title": title,
        "body": body,
        "assignees": [assignee]
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    print("🆕 Created issue:", resp.json()["html_url"])

def build_failure_body():
    globus_log = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{BATCH_ID}.log"
    return (
        f"### Task `{TASK_NAME}` failed for batch `{BATCH_ID}`\n\n"
        f"**Error message:**\n```\n{ERROR_MSG}\n```\n\n"
        f"[Click here to view the log]({globus_log})\n\n"
        f"_Assigned to @{ASSIGNEE}_"
    )

def build_success_body():
    globus_pdf = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{BATCH_ID}_report.pdf"
    globus_log = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{BATCH_ID}.log"
    return (
        "### New Inspection Report Available\n\n"
        f"The inspection report for batch `{BATCH_ID}` is ready for review.\n\n"
        f"[View Report PDF]({globus_pdf})\n"
        f"[View Log File]({globus_log})\n\n"
        f"_Assigned to @{ASSIGNEE}_\n\n"
        "---\n"
        "#### Manual Review Instructions\n"
        "| Type | Description |\n"
        "|------|-------------|\n"
        "| Preprocessing Quality | Artifacts, exposure, color correction issues |\n"
        "| Potting Area Cleanliness | Messy with lots of residue or cluttered potting area |\n"
        "| Non-Target Weeds | Presence of unwanted weeds in the pots or on the landscape fabric |\n"
        "| Plant Spacing | Plants too close or overlapping |\n"
        "| Species Labeling | Species bounding box has been mislabeled |\n"
        "| Area (cm²) | Area estimation seems wrong |\n"
        "| Reconstruction Issues | Scene looks distorted or incomplete |\n"
        "| Other | Any other concerns |\n"
    )

def main():
    issue_number = find_existing_issue(BATCH_ID)
    body = build_success_body() if IS_SUCCESS else build_failure_body()
    title = f"{BATCH_ID}: Inspection Report" if IS_SUCCESS else f"{BATCH_ID}: Task Failed - {TASK_NAME}"

    if issue_number:
        comment_on_issue(issue_number, body)
    else:
        create_issue(title, body, ASSIGNEE)

if __name__ == "__main__":
    main()
