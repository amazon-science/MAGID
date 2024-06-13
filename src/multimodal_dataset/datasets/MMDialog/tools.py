from io import BytesIO
from PIL import Image
import json
from base64 import b64decode
import base64
import os
import re
import requests
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoProcessor, CLIPVisionModel, CLIPTextModel
from multimodal_dataset.graphics.chat_vis import visualize_conversation

import boto3
client = boto3.client('rekognition', region_name='us-west-2')


def remove_emoji(text):
    return re.sub(":.*?:", "", text)


def prepare_content(chat):
    assert type(chat) == list and len(chat) > 0
    content = ""
    content2 = []
    for i, element in enumerate(chat):
        temp_content = ""
        for utternece in element["turn"]:
            if ("__TEXT__" in utternece):
                temp_content += remove_emoji(utternece["__TEXT__"]) + " "
                # temp_content += utternece["__TEXT__"] + " "
        if (i == len(chat) - 1):
            content += "Utterance: " + str(i) + ": " + temp_content.strip()
        else:
            content += "Utterance: " + \
                str(i) + ": " + temp_content.strip() + " \n "
        content2.append(temp_content.strip())
    return content, content2


def get_prompts(answer):
    assert type(answer) == str
    prompts = []
    if ("<result>" not in answer):
        return ""
    filtered_answer = re.findall(
        r'<result>(.*?)</result>', answer, re.DOTALL)[0]
    for i, element in enumerate(filtered_answer.split("\n")):
        if (len(element) <= 4 or re.search(r'\d+', element) is None):
            continue
        index = re.search(r'\d+', element).group()
        prompts.append([index, element[element.index(":") + 1:]])
    return prompts

# Visualize the conversation for MMDialgoue dataset


def visulaize_conversation_mmdialogue(list_conversation, image_path_MMDialog, utterances_image_magi,
                                      image_gen_magi, MM_relevance_images, target_label, folder_path_results, num_images):
    vis_conversation = []
    image_text_pairs = []
    for i, element in enumerate(list_conversation):
        text_utterence = ""
        flag_media = False
        for utternece in element["turn"]:
            if ("__TEXT__" in utternece):
                vis_conversation.append(
                    ("txt", remove_emoji(utternece["__TEXT__"])))
                text_utterence += remove_emoji(utternece["__TEXT__"])
            elif ("__MEDIA__" in utternece):
                img_target_path = os.path.join(
                    image_path_MMDialog, utternece["__MEDIA__"] + ".jpg")
                vis_conversation.append(("img", img_target_path))
                image_text_pairs.append((img_target_path, i))
                num_images[0] += 1
                flag_media = True
        if (flag_media):
            target_label.append(1)
        else:
            target_label.append(0)
        print("Utterance: {}: {} ".format(i, text_utterence))
    visualize_conversation(vis_conversation, os.path.join(
        folder_path_results, "vis.png"))
    return image_text_pairs
