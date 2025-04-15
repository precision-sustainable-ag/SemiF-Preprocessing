import logging
from datetime import datetime
from pathlib import Path

import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

log = logging.getLogger(__name__)

def build_https_globus_link(file_path: str, http_root_url: str, local_mount_prefix: str) -> str:
    """
    Builds a Globus HTTPS-accessible link from a local file path.

    Args:
        file_path (str): Full path to file (e.g., /mnt/data/rsstu/users/.../file.pdf)
        http_root_url (str): Root HTTP URL (e.g., https://m-ae9b45.7ce1a.03c0.data.globus.org/)
        local_mount_prefix (str): Local base path mapped to the HTTP root (e.g., /mnt/data)

    Returns:
        str: A complete HTTP-accessible Globus link
    """
    rel_path = Path(file_path).resolve().relative_to(Path(local_mount_prefix).resolve())
    return f"{http_root_url.rstrip('/')}/{rel_path.as_posix()}"

def read_yaml(path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")
    
def generate_summary_message(message, user_id=None, message_type: str ="Error", task_instruction: str = None):
        mention_text = f"<@{user_id}>\n" if user_id else ""
        instruction_text = f"\n\n{task_instruction}" if task_instruction else ""

        message_blocks = [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*SemiF-PreProcessing {message_type} Message* - "
                            f"{datetime.now().strftime('%m/%d/%Y')}"
                }
            },
            {
                'type': 'divider'
            }
        ]
        
        message_blocks.append({
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f"{mention_text}{message}{instruction_text}"
            }
        })
        return message_blocks

def send_slack_notification(cfg, message_blocks, files):
        client = WebClient(token=read_yaml(cfg.paths.pipeline_keys)["slack_token"])
        message = {
            'channel': cfg.slack_channel,
            'blocks': message_blocks
        }
        try:
            message_response = client.chat_postMessage(**message)
            for file in files:
                client.files_upload_v2(
                        channel=message_response['channel'],
                        file=file,
                        # initial_comment="Here's the attached file",
                        thread_ts=message_response['ts'],
                    )

            log.info(
                f"sent slack message to channel - {message_response['channel']}, thread - {message_response['ts']}")
        except SlackApiError as e:
            print(f"Error: {e}")