import re
import os

from multimodal_dataset.graphics.chat_vis import visualize_conversation


def prepare_content_MMDD(chat):
    assert type(chat) == list and len(chat) > 0
    content = ""
    for i, element in enumerate(chat):
        if (i == len(chat) - 1):
            content += "Utterance: " + str(i) + ": " + element
        else:
            content += "Utterance: " + str(i) + ": " + element + " \n "
    return content, chat


def get_prompts(answer):
    assert type(answer) == str
    prompts = []
    for i, element in enumerate(answer.split("\n")):
        if(len(element) <= 4):
            continue
        index = re.search(r'\d+', element).group()
        prompts.append([index, element[element.index(":") + 1:]])
    return prompts


def visulaize_conversation_MMDD(df, image_path_MMDD, utterances_image_magi,
                                image_gen_magi, MM_relevance_images, target_label, folder_path_results, index, num_images):
    image_text_pairs = []
    list_conversation = df["dialog"].iloc[index]
    index_replace = df["replaced_idx"].iloc[index]
    image_mmdd = df["img_file"][index]
    if ("COCO" not in image_mmdd):
        raise Exception("Image not found for MMDD dataset")
    vis_conversation = []
    for i, element in enumerate(list_conversation):
        vis_conversation.append(("txt", element))
        if (i == index_replace):
            img_target_path = os.path.join(image_path_MMDD, image_mmdd)
            vis_conversation.append(("img", img_target_path))
            image_text_pairs.append((img_target_path, i))
            num_images[0] += 1
            target_label.append(1)
        else:
            target_label.append(0)
        print("Utterance: {}: {} ".format(i, element))
    visualize_conversation(vis_conversation, os.path.join(
        folder_path_results, "vis.png"))
    return image_text_pairs
