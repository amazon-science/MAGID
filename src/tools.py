from io import BytesIO
import yaml
import pickle as pkl
import openai
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, CLIPVisionModel, CLIPTextModel
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, f1_score
import openai
import os
import torch
import numpy as np
import random
import json
import logging
logger = logging.getLogger(__name__)


def print_config(config):
    logging.warning("Configuration of the running experiment:")
    config_dict = vars(config)
    logger.warning(json.dumps(config_dict, indent=4, sort_keys=True))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # create a generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator


def evaluate(target_label, model_label):

    # precision
    precision = precision_score(target_label, model_label)

    # recall
    recall = recall_score(target_label, model_label)

    # accuracy
    accuracy = accuracy_score(target_label, model_label)

    # f1 score
    F1 = f1_score(target_label, model_label)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_label, model_label).ravel()
    conf_matrix = np.array([[tp, fp], [fn, tn]])
    return "Accuracy: {} | Precsion: {} | Recall: {} | F1: {} \n Confusion Matrix: \n {}".format(accuracy, precision, recall, F1, conf_matrix)


def truncate_text(list_text, max_tokens=75):
    """
    Truncate text to ensure it does not exceed max tokens when tokenized.
    """
    shorter_list_text = []
    for text in list_text:
        # This is a simple way to tokenize by spaces; consider using a more advanced tokenizer if needed
        tokens = text.split()

        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]  # Truncate the tokens

        # Join the tokens back into a string
        shorter_list_text.append(' '.join(tokens))
    return shorter_list_text


def get_dataset_config(dataset_name, config_yaml):
    with open(config_yaml, 'r') as file:
        data = yaml.safe_load(file)
    datasets = data["datasets"]
    return datasets[dataset_name]["path_conversation"], datasets[dataset_name]["path_images"]


def set_credentials_config(credentials_yaml):
    with open(credentials_yaml, 'r') as file:
        data = yaml.safe_load(file)
    os.environ["OPENAI_API_KEY"] = data["openai"]
    openai.api_key = data["openai"]
    os.environ['STABILITY_HOST'] = data["stability_ai"]["host"]
    os.environ['STABILITY_KEY'] = data["stability_ai"]["key"]
    print("Credentials are set.")


def get_candidate_indices(df, dataset_name, path=None, size_experiment=-1, split=-1, use_saved_ids=False):
    def split_list_to_four(input_list):
        n = len(input_list)
        base_size, remainder = divmod(n, 4)

        result = []
        start = 0
        for _ in range(4):
            size = base_size + (remainder > 0)
            result.append(input_list[start:start+size])
            start += size
            remainder -= 1
        return result
    if (use_saved_ids is False):
        candidate_indices = random.sample(range(len(df)), size_experiment)
        new_indces = []
        # if(dataset_name == "MMDD"):
        #     for index in candidate_indices:
        #       image_mmdd = df["img_file"][index]
        #       if ("COCO" in image_mmdd):
        #             new_indces.append(index)
        #     candidate_indices = new_indces
    elif(split != -1):
        with open(path, 'rb') as file:
            data = pkl.load(file)
        list_splits = split_list_to_four(data)
        candidate_indices = list_splits[split]
    elif(use_saved_ids is True and split == -1):
        if os.path.exists(path) is False:
            candidate_indices = list(random.sample(
                range(len(df)), size_experiment))
            with open(path, "wb") as file:
                pkl.dump(candidate_indices, file)
        with open(path, 'rb') as file:
            candidate_indices = pkl.load(file)
    return candidate_indices
