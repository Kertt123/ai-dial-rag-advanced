import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str, dimensions: int = 1536):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.dimensions = dimensions
        self.url = DIAL_EMBEDDINGS.format(model=deployment_name)

    def get_embeddings(self, input_list: list[str]) -> dict[int, list[float]]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        payload = {
            "input": input_list,
            "dimensions": self.dimensions
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        embeddings_dict = {}
        for item in result["data"]:
            embeddings_dict[item["index"]] = item["embedding"]

        return embeddings_dict

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
