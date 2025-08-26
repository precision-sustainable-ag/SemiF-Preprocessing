import os
import requests

GITHUB_API = "https://api.github.com"
TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["REPO"]
BATCH_ID = os.environ["BATCH_ID"]
START_TIME = os.environ.get("START_TIME", "").strip()
ASSIGNEE = os.environ["ASSIGNEE"]
GLOBUS_PREFIX = os.environ["GLOBUS_PREFIX"]

IS_SUCCESS = os.environ.get("IS_SUCCESS", "false").lower() == "true"
TASK_NAME = os.environ.get("TASK_NAME", "Unknown")
ERROR_MSG = os.environ.get("ERROR_MSG", "No message provided")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def remove_label(issue_number, label):
    url = f"{GITHUB_API}/repos/{REPO}/issues/{issue_number}/labels/{label}"
    resp = requests.delete(url, headers=HEADERS)
    if resp.status_code != 204:
        print(f"Could not remove label '{label}':", resp.text)

def add_label(issue_number, label):
    url = f"{GITHUB_API}/repos/{REPO}/issues/{issue_number}/labels"
    resp = requests.post(url, json={"labels": [label]}, headers=HEADERS)
    if resp.status_code not in [200, 201]:
        print(f"Could not add label '{label}':", resp.text)

def find_existing_issue(batch_id, exact_title):
    url = f"{GITHUB_API}/repos/{REPO}/issues"
    page = 1

    while True:
        resp = requests.get(f"{url}?state=open&per_page=100&page={page}", headers=HEADERS)
        resp.raise_for_status()
        issues = resp.json()

        if not issues:
            break

        for issue in issues:
            if issue.get("title") == exact_title:
                return issue["number"]

        page += 1

    return None

def comment_on_issue(issue_number, message):
    url = f"{GITHUB_API}/repos/{REPO}/issues/{issue_number}/comments"
    payload = {"body": message}
    resp = requests.post(url, json=payload, headers=HEADERS)
    resp.raise_for_status()
    print(f"Commented on issue #{issue_number}")

def create_issue(title, body, assignee):
    url = f"{GITHUB_API}/repos/{REPO}/issues"
    payload = {
        "title": title,
        "body": body,
        "assignees": [assignee]
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    print("Created issue:", resp.json()["html_url"])

def build_failure_body():
    globus_log = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}.log"
    return (
        f"### Task `{TASK_NAME}` failed for batch `{BATCH_ID}`\n\n"
        f"**Error message:**\n```\n{ERROR_MSG}\n```\n\n"
        f"[Click here to view the log]({globus_log})\n\n"
        f"_Assigned to @{ASSIGNEE}_"
    )

def build_non_first_success_body():
    globus_pdf = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}_{START_TIME}_report.pdf"
    globus_asfm_pdf = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}_asfm_report.pdf"
    globus_log = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}.yaml"
    return (
        "### New Inspection Report Available\n\n"
        f"The inspection report for batch `{BATCH_ID}` is ready for review.\n\n"
        f"[View Report PDF]({globus_pdf})\n"
        f"[View ASFM Report PDF]({globus_asfm_pdf})\n"
        f"[View Log File]({globus_log})\n\n"
        f"_Assigned to @{ASSIGNEE}_\n\n"
    )

def build_first_success_body():
    globus_pdf = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}_{START_TIME}_report.pdf"
    globus_asfm_pdf = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}_asfm_report.pdf"
    globus_log = f"{GLOBUS_PREFIX}/{BATCH_ID}/inspection/{START_TIME}/{BATCH_ID}.yaml"
    return (
        "### New Inspection Report Available\n\n"
        f"The inspection report for batch `{BATCH_ID}` is ready for review.\n\n"
        f"[View Report PDF]({globus_pdf})\n"
        f"[View ASFM Report PDF]({globus_asfm_pdf})\n"
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
    
    body = build_first_success_body() if IS_SUCCESS else build_failure_body()
    # Compose unique title per (batch_id, start_time)
    start_suffix = f" [Start {START_TIME}]" if START_TIME else ""
    title = f"{BATCH_ID}{start_suffix} Batch Processing Request"
    issue_number = find_existing_issue(BATCH_ID, title)

    if issue_number:
        body = build_non_first_success_body() if IS_SUCCESS else body
        comment_on_issue(issue_number, body)
    
        # Modify labels based on status
        if IS_SUCCESS:
            remove_label(issue_number, "bug")
            add_label(issue_number, "fixed")
        else:
            add_label(issue_number, "bug")
    else:
        create_issue(title, body, ASSIGNEE)
        issue_number = find_existing_issue(BATCH_ID, title)  # Needed to get issue number for label ops
        if IS_SUCCESS:
            add_label(issue_number, "completed")
        else:
            add_label(issue_number, "bug")


if __name__ == "__main__":
    main()
