import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests

from client import Client
from text_processing import TextProcessing as tp


class PromptProcessor:
    """Processes prompts and handles LLM responses."""

    def __init__(self, client: Client):
        """Initializes PromptProcessor."""
        self.client = client

    def process_prompt(self, prompt: str) -> Optional[str]:
        """Sends prompt to LLM server and returns response."""
        try:
            start_time = time.perf_counter()
            result = self.client.generate_text(prompt)
            end_time = time.perf_counter()

            if "text" in result:
                response = result["text"]
            elif "detail" in result:
                response = result["detail"]
            else:
                raise ValueError("Unexpected LLM response format")

            if not response:
                raise ValueError("Empty LLM response content")

            response = tp.clean_mistral(response)

            tps = tp.measure_performance(start_time, end_time, response)
            print(f"Tokens per second: {tps} t/s")
            return response

        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"An error occurred: {e}")
            return None


def main():
    """Conversation loop with LLM server."""
    client = Client()
    processor = PromptProcessor(client)

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ["quit", "exit"]:
            print("Exiting the conversation.")
            break

        response = processor.process_prompt(prompt)
        print(f"{response}")


if __name__ == "__main__":
    main()
