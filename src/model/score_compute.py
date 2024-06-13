from transformers import CLIPProcessor, CLIPModel
import torch
# from model.nsfw_detect import NSFW_detect
from model.vision_qa.aesthetic_predictor import AestheticPredictor


class Score_compute:
    def __init__(self) -> None:
        self.model_clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.processor_clip = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.aesthetic_predictor = AestheticPredictor()
        
        # add your nfsw-detection code here. Example:
        # self.nfsw_model_detect = NSFW_detect()

    def compute_clip_score(self, clip_content, image):
        with torch.no_grad():
            inputs_clip = self.processor_clip(
                text=clip_content, images=image, return_tensors="pt", padding=True)
            outputs_clip = self.model_clip(**inputs_clip)
            # scores = outputs_clip.logits_per_image.softmax(dim=1)
            scores = outputs_clip.logits_per_image.item()/100
        return scores

    def aesthetic_score_compute(self, image):
        with torch.no_grad():
            score_aesthetic = self.aesthetic_predictor.predict(image)/10
        return score_aesthetic

    #Example nsfw-detection code. 
    # def nsfw_detect(self, image):
    #     prediction = self.nfsw_model_detect.predict(image)
    #     return prediction
