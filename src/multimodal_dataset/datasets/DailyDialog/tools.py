import re
import os

from multimodal_dataset.graphics.chat_vis import visualize_conversation


def prepare_content_DialyDialog(chat):
    assert type(chat) == list and len(chat) > 0
    content = ""
    for i, element in enumerate(chat):
        if (i == len(chat) - 1):
            content += "Utterance: " + str(i) + ": " + element
        else:
            content += "Utterance: " + str(i) + ": " + element + " \n "
    return content, chat


def visulaize_conversation_DailyDialog(df, image_path_DailyDialog, utterances_image_magi,
                                       image_gen_magi, MM_relevance_images, target_label, folder_path_results, index):
    image_text_pairs = []
    list_conversation = df[index]
    vis_conversation = []
    for i, element in enumerate(list_conversation):
        vis_conversation.append(("txt", element))
        target_label.append(0)
        print("Utterance: {}: {} ".format(i, element))
    visualize_conversation(vis_conversation, os.path.join(
        folder_path_results, "vis.png"))
    return image_text_pairs
