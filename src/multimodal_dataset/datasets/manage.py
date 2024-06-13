import pandas as pd
import os
import torch
from PIL import Image
from multimodal_dataset.datasets.MMDD.tools import prepare_content_MMDD, visulaize_conversation_MMDD
from multimodal_dataset.datasets.MMDialog.tools import prepare_content, visulaize_conversation_mmdialogue
from multimodal_dataset.datasets.DailyDialog.tools import prepare_content_DialyDialog, visulaize_conversation_DailyDialog
from multimodal_dataset.datasets.photochat.tools import prepare_content_photochat, visulaize_conversation_photochat


class DatasetManage:

    def __init__(self, dataset_name, path_dataset, score_compute) -> None:
        self.dataset_name = dataset_name
        self.df = self.read_dataset(path_dataset)
        self.score_compute = score_compute

    def read_dataset(self, path_dataset):
        if (self.dataset_name == "MMDialog"):
            return pd.read_json(path_dataset, lines=True)["conversation"]
        elif (self.dataset_name == "MMDD"):
            return pd.read_json(path_dataset)
        elif (self.dataset_name == "PhotoChat"):
            return pd.read_json(path_dataset)
        elif (self.dataset_name == "DailyDialog"):
            # Open the file
            with open(path_dataset, 'r') as file:
                # Read the file
                data = file.read()
            rows = data.split("\n")
            data_list = [row.split('__eou__') for row in rows]
            return data_list
        elif (self.dataset_name == "PersonaChat"):
            df = pd.read_csv(path_dataset)
            return df
        else:
            raise Exception("Dataset not implemented")

    def prep_content(self, index):
        if (self.dataset_name == "MMDialog"):
            list_conversation = self.df.iloc[index]
            llm_content, clip_content = prepare_content(list_conversation)
            return llm_content, clip_content, list_conversation
        elif (self.dataset_name == "MMDD"):
            list_conversation = self.df.iloc[index]["dialog"]
            llm_content, clip_content = prepare_content_MMDD(list_conversation)
            return llm_content, clip_content, list_conversation
        elif (self.dataset_name == "PhotoChat"):
            list_conversation = self.df.iloc[index]["dialogue"]
            llm_content, clip_content, _ = prepare_content_photochat(
                list_conversation)
            return llm_content, clip_content, list_conversation
        elif (self.dataset_name == "DailyDialog"):
            list_conversation = self.df[index]
            llm_content, clip_content = prepare_content_DialyDialog(
                list_conversation)
            return llm_content, clip_content, list_conversation
        elif (self.dataset_name == "PersonaChat"):
            list_conversation = self.df["chat"].iloc[index].split("\n")
            list_conversation.pop()
            llm_content, clip_content = prepare_content_DialyDialog(
                list_conversation)
            return llm_content, clip_content, list_conversation

    def compute_clip_score(self, image_text_pairs, clip_content, clip_s_sec):
        if(len(image_text_pairs) == 0):
            return
        torch.cuda.empty_cache()
        assert type(image_text_pairs) == list
        assert type(image_text_pairs[0]) == tuple
        assert type(image_text_pairs[0][0]) == str
        assert type(image_text_pairs[0][1]) == int
        for pairs in image_text_pairs:
            image_path, index = pairs
            image = Image.open(image_path)
            scores = self.score_compute.compute_clip_score(
                clip_content[index], image)
            clip_s_sec.append(scores)

    def compute_aesthetic_score(self, image_text_pairs, aesthetic_sec):
        for pairs in image_text_pairs:
            image_path, index = pairs
            image = Image.open(image_path)
            score_aesthetic = self.score_compute.aesthetic_score_compute(image)
            aesthetic_sec.append(score_aesthetic)
