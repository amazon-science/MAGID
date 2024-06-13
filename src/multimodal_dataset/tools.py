from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, f1_score
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
    conf_matrix = confusion_matrix(target_label, model_label)
    return "Accuracy: {} | Precsion: {} | Recall: {} | F1: {} \n Confusion Matrix: {}".format(accuracy, precision, recall, F1, conf_matrix)
