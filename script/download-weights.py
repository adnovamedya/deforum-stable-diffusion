#!/usr/bin/env python

import os
import sys
import shutil
from tqdm import tqdm
import requests


MODEL_CACHE = "models"
# if os.path.exists(MODEL_CACHE):
#    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

MODEL_MAP = {
    "Protogen_V2.2.ckpt": {
        "sha256": "bb725eaf2ed90092e68b892a1d6262f538131a7ec6a736e50ae534be6b5bd7b1",
        "url": "https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2.ckpt",
        "requires_login": False,
    }
}

WEIGHTS = [
    "https://huggingface.co/deforum/AdaBins/resolve/main/AdaBins_nyu.pt",
    "https://huggingface.co/deforum/MiDaS/resolve/main/dpt_large-midas-2f21e586.pt"
]

def download_model(model_ckpt):
    if (os.path.exists(os.path.join(MODEL_CACHE, model_ckpt))):
        print(f"Model {model_ckpt} already downloaded")
        return

    url = MODEL_MAP[model_ckpt]["url"]
    if MODEL_MAP[model_ckpt]["requires_login"]:
        username = sys.argv[1]
        token = sys.argv[2]
        _, path = url.split("https://")
        url = f"https://{username}:{token}@{path}"

    # contact server for model
    print(f"..attempting to download {model_ckpt}...this may take a while")
    ckpt_request = requests.get(url, stream=True)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError(
            "You have not accepted the license for this model."
        )
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(
            f"Some other error has ocurred - response code: {request_status}"
        )

    # write to model path
    with open(os.path.join(MODEL_CACHE, model_ckpt), "wb") as model_file:
        file_size = int(ckpt_request.headers.get("Content-Length"))
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=model_ckpt) as pbar:
            for chunk in ckpt_request.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    model_file.write(chunk)
                    pbar.update(len(chunk))


def download_file(url, models_path):
    filename = url.split("/")[-1]

    if (os.path.exists(os.path.join(models_path, filename))):
        print(f"File {filename} already downloaded")
        return

    print(f"..attempting to download {filename}...this may take a while")
    # Create the models_path directory if it does not exist
    os.makedirs(models_path, exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Get the total file size
    file_size = int(response.headers.get("Content-Length"))

    # Open a file in binary mode to write the content
    with open(os.path.join(models_path, filename), "wb") as f:
        # Initialize the progress bar
        pbar = tqdm(total=file_size, unit="B", unit_scale=True)

        # Iterate through the response data and write it to the file
        for data in response.iter_content(1024):
            f.write(data)
            # Update the progress bar manually
            pbar.update(len(data))

        # Close the progress bar
        pbar.close()

# download checkpoints
for model_ckpt in MODEL_MAP:
    download_model(model_ckpt)

for weight_url in WEIGHTS:
    download_file(weight_url, MODEL_CACHE)