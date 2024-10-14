import labelbox
import os
import json
import numpy as np
from PIL import Image
from io import BytesIO
import requests

def getJSON():

    client = labelbox.Client(api_key = LB_API_KEY)
    export_task = labelbox.ExportTask.get_task(client, "cm0i2ndbl0h1h071tdvya5jpw") # cm0fl037n00v5070mc55m42fn
    output_dir = "Iris 300/json_masks"
    os.makedirs(output_dir, exist_ok = True)
    file_counter = 0

    def jsonStreamHandler(output: labelbox.JsonConverterOutput):
        nonlocal file_counter
        file_counter += 1
        file_path = os.path.join(output_dir, f"output_{file_counter}.json")
        with open(file_path, "w") as f:
            json.dump(json.loads(output.json_str), f, indent = 2)
        print(f"Saved to {file_path}")
    export_task.get_stream().start(stream_handler = jsonStreamHandler)

def makePNG():
    json_folder_path = "Iris 300/json_masks"
    output_dir_mask = "Iris 300/SegmentationClass"
    output_dir_original = "Iris 300/OriginalData"

    if not os.path.exists(output_dir_mask):
        os.makedirs(output_dir_mask)
    if not os.path.exists(output_dir_original):
        os.makedirs(output_dir_original)

    for json_file_name in os.listdir(json_folder_path):
        if json_file_name.endswith(".json"):
            json_file_path = os.path.join(json_folder_path, json_file_name)
            with open(json_file_path, "r") as file:
                json_data = json.load(file)
            external_id = json_data["data_row"]["external_id"].replace(".jpg", "")
            mask_url = None
            for project_id, project_data in json_data["projects"].items():
                for label in project_data.get("labels", []):
                    for obj in label["annotations"]["objects"]:
                        mask_url = obj["mask"].get("url")
                        if mask_url:
                            break
                    if mask_url:
                        break
                if mask_url:
                    break
            try:
                original_url = json_data["data_row"]["row_data"]
                response_original = requests.get(original_url)
                response_original.raise_for_status()
                img_original = Image.open(BytesIO(response_original.content))
                original_output_path = os.path.join(output_dir_original, f"{external_id}.png")
                img_original.save(original_output_path)
                print(f"Saved original image to {original_output_path}")
                if mask_url:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    response_mask = requests.get(mask_url, headers = headers)
                    response_mask.raise_for_status()
                    img_mask = Image.open(BytesIO(response_mask.content))
                else:
                    img_mask = Image.new("L", img_original.size, color = 0)
                    print(f"Created black mask for {external_id}")
                mask_output_path = os.path.join(output_dir_mask, f"{external_id}_mask.png")
                img_mask.save(mask_output_path)
                print(f"Saved mask image to {mask_output_path}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image or mask from {json_file_name}: {e}")