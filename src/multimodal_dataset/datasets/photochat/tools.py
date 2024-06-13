import re
import os
import requests

from multimodal_dataset.graphics.chat_vis import visualize_conversation


def prepare_content_photochat(chat):
    assert type(chat) == list and len(chat) > 0
    content = ""
    final_chat = ""
    previous_user = None
    counter = 0
    image_index = 0
    for i, element in enumerate(chat):
        if(previous_user != element["user_id"]):
            content += "Utterance: " + \
                str(counter) + ": " + element['message'] + " \n "
            final_chat += element['message'] + " \n "
            if(element["share_photo"] is True):
                image_index = counter
            counter += 1
            previous_user = element["user_id"]
            continue
        else:
            content = content[:-3] + " " + element['message'] + " \n "
            final_chat = final_chat[:-3] + " " + element['message'] + " \n "
        previous_user = element["user_id"]
        if(element["share_photo"] is True):
            image_index = counter-1
    print(content[:-3])
    return content[:-3], final_chat[:-3].split("\n"), image_index


def download_and_save_image(url, filename):
    # Check if the image already exists
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return filename

    # If not, proceed to download
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Open the image file in binary-write mode and save it
    with open(filename, 'wb') as file:
        # write in chunks to handle large files
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {url} and saved as {filename}")

