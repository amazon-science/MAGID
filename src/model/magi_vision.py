from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline
from pathlib import Path
import torch
import time
from stability_sdk import client
import os
from huggingface_hub import login
import io
import warnings
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


class MAGI_vision:

    def __init__(self, diffusion_type, max_retry=2, min_score_aesthetic=0.51, score_compute=None) -> None:
        self.repeat_dreamstudio = 0
        self.retry = 0
        self.score_compute = score_compute
        self.min_score_aesthetic = min_score_aesthetic
        self.max_retry = max_retry
        self.diffusion_type = diffusion_type
        self.diffusion_model = self.construct_diffusion_model()

    def construct_diffusion_model(self):
        if self.diffusion_type == "stabilityai/stable-diffusion-2":
            scheduler = EulerDiscreteScheduler.from_pretrained(
                self.diffusion_type, subfolder="scheduler")
            pipe = StableDiffusionPipeline.from_pretrained(
                self.diffusion_type, scheduler=scheduler, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            return pipe
        elif self.diffusion_type == "stabilityai/stable-diffusion-xl-base-0.9":
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            except:
                login()
                pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe = pipe.to("cuda")
            return pipe
        elif self.diffusion_type == "stabilityai/stable-diffusion-xl-base-1.0":
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            except:
                login()
                pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe = pipe.to("cuda")
            return pipe
        elif self.diffusion_type == "dreamstudio":
            stability_api = client.StabilityInference(
                key=os.environ['STABILITY_KEY'],  # API Key reference.
                verbose=True,  # Print debug messages.
                # Set the engine to use for generation.
                engine="stable-diffusion-xl-beta-v2-2-2",
                # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
                # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
            )
            return stability_api

    def get_diffusion_model_image(self, prompt, seed=120):
        if self.diffusion_type == "stabilityai/stable-diffusion-2":
            return self.diffusion_model(prompt).images[0]
        elif self.diffusion_type == "stabilityai/stable-diffusion-xl-base-0.9":
            return self.diffusion_model(prompt=prompt).images[0]
        elif self.diffusion_type == "stabilityai/stable-diffusion-xl-base-1.0":
            return self.diffusion_model(prompt=prompt).images[0]
        elif self.diffusion_type == "dreamstudio":
            answers = self.diffusion_model.generate(
                prompt=prompt,
                # If a seed is provided, the resulting generated image will be deterministic.
                seed=seed,
                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
                # Amount of inference steps performed on image generation. Defaults to 30.
                steps=30,
                # Influences how strongly your generation is guided to match your prompt.
                cfg_scale=8.0,
                # Setting this value higher increases the strength in which it tries to match your prompt.
                # Defaults to 7.0 if not specified.
                # Generation width, defaults to 512 if not included.
                width=512,
                # Generation height, defaults to 512 if not included.
                height=512,
                # Number of images to generate, defaults to 1 if not included.
                samples=1,
                # Choose which sampler we want to denoise our generation with.
                sampler=generation.SAMPLER_K_DPMPP_2M
                # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
            )
            # Set up our warning to print to the console if the adult content classifier is tripped.
            # If adult content classifier is not tripped, save generated images.
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again.")
                        if (self.repeat_dreamstudio < 2):
                            self.repeat_dreamstudio += 1
                            return self.get_diffusion_model_image(prompt, seed=seed + 1)
                        else:
                            return None
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        self.repeat_dreamstudio = 0
                        return img

    def generate_response(self, prompt, clip_content, index_utterance, seed=120, path_save_image=None):
        if (Path(path_save_image).exists() and False):
            generated_image = Image.open(path_save_image)
        else:
            with torch.no_grad():
                generated_image = self.get_diffusion_model_image(
                    self.replace_last_dot(prompt, ", photographic style."), seed)
        torch.cuda.empty_cache()
        with torch.no_grad():
            scores = self.score_compute.compute_clip_score(
                clip_content[index_utterance], generated_image)
        return generated_image, scores

    def reset_retry(self):
        self.retry = 0

    def replace_last_dot(self, s, r):
        return s[:s.rfind('.')] + r + s[s.rfind('.')+1:] if '.' in s else s + r

    def generate_image(self, prompt, index_utterance, clip_content, image_path, threshold, seed=120, logger=None):
        torch.cuda.empty_cache()
        if(self.retry > self.max_retry):
            return None, None, None, None
        path_save_image = os.path.join(image_path, "image_" + str(
            seed) + "_" + prompt[:20].replace(" ", "_").replace("/", "_") + ".png")
        image, score_clip = self.generate_response(
            prompt, clip_content, index_utterance, seed=seed, path_save_image=path_save_image)
        score_aesthetic = self.score_compute.aesthetic_score_compute(image)
        logger.info("Aesthetic score for image {} is: {}".format(
            path_save_image, str(score_aesthetic)))

        image.save(path_save_image)
        
        # add your nfsw-detection code here. Example:
        # score_safe, class_safe = self.score_compute.nsfw_detect(image)
        # logger.info("NSFW class of '{}' is set for image: {}".format(
        #     str(class_safe), path_save_image))
        # if(score_safe == False):
        #     self.retry += 1
        #     logger.info("Not safe for image {}.".format(path_save_image))
        #     return self.generate_image(prompt, index_utterance, clip_content, image_path, threshold, seed=seed + 1, logger=logger)
        
        if(score_aesthetic < self.min_score_aesthetic):
            self.retry += 1
            logger.info("Low aesthetic score is too low {} for image {}".format(
                str(score_aesthetic), path_save_image))
            return self.generate_image(prompt, index_utterance, clip_content, image_path, threshold, seed=seed + 1, logger=logger)
        if score_clip < threshold and self.retry < self.max_retry:
            self.retry += 1
            logger.info("Low CLIP score  {}| prompt: {} | utterance index: {}".format(
                str(score_clip), prompt, str(index_utterance)))
            return self.generate_image(prompt, index_utterance, clip_content, image_path, threshold, seed=seed + 1, logger=logger)
        elif(self.retry >= self.max_retry and score_clip < threshold):
            image = None
        else:
            self.retry = 0
        return image, score_clip, score_aesthetic, path_save_image
