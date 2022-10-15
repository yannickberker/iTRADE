"""Upload data to Zenodo."""
import json
import os
import re
from pathlib import Path

import markdown2
import requests

from itrade import DATA_DIR, DATA_ZIP_NAME

ENDPOINT = "https://zenodo.org/api/deposit/depositions"


def zenodo_upload() -> None:
    """Upload data to Zenodo."""
    params = {"access_token": os.getenv("ZENODO_PAT")}

    # Get entries
    r = requests.get(ENDPOINT, params=params)
    if r.status_code != 200:
        print(r.json())
        raise AssertionError

    if depositions := r.json():
        if len(depositions) > 1:
            raise AssertionError
        deposition = depositions[0]
    else:
        # Add an entry
        r = requests.post(ENDPOINT, params=params, json={})
        if r.status_code != 201:
            print(r.json())
            raise AssertionError
        deposition = r.json()

    deposition_id = deposition["id"]
    prereserved_doi = deposition["metadata"]["prereserve_doi"]["doi"]
    print(f"Deposition {deposition_id}, DOI {prereserved_doi}")

    # Add metadata
    metadata_url = f"{ENDPOINT}/{deposition_id}"
    metadata_str = (Path(__file__).parent.parent / ".zenodo.data.json").read_text()
    metadata = json.loads(metadata_str)

    readme = (DATA_DIR / "README.md").read_text()
    readme = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", readme)

    description = markdown2.markdown(readme, extras=["fenced-code-blocks"])
    description = re.sub("<(/?)h1>", r"<\1code>", description)
    description = re.sub("<(/?)h2>", r"<\1b>", description)
    description = re.sub("<(/?)h3>", r"<\1i>", description)
    metadata["description"] = description

    r = requests.put(metadata_url, params=params, json={"metadata": metadata})
    if r.status_code != 200:
        print(r.json())
        raise AssertionError

    # Upload file
    if not (DATA_DIR / DATA_ZIP_NAME).is_file():
        print("Done without upload.")
        return

    bucket_url = deposition["links"].get("bucket")
    if bucket_url is None:
        bucket_url = os.getenv(f"ZENODO_BUCKET_URL_{deposition_id}")
    if bucket_url is None:
        # Old API
        upload_url = f"{metadata_url}/files"
        data = {"name": DATA_ZIP_NAME}
        with open(DATA_DIR / DATA_ZIP_NAME, "rb") as file:
            files = {"file": file}
            r = requests.post(upload_url, params=params, data=data, files=files)
        if r.status_code != 201:
            print(r.json())
            raise AssertionError
        print("Done with upload via old API.")
        return

    # New API
    upload_url = bucket_url + "/" + DATA_ZIP_NAME
    with open(DATA_DIR / DATA_ZIP_NAME, "rb") as file:
        r = requests.put(upload_url, params=params, data=file)
    if r.status_code != 200:
        print(r.json())
        raise AssertionError

    print("Done with upload via new API.")


if __name__ == "__main__":
    zenodo_upload()
