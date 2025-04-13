import logging
from datetime import datetime

import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

log = logging.getLogger(__name__)

def read_yaml(path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")
    
def generate_summary_message(message, message_type="Error"):
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
                'text': message
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