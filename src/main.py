from model.prompt import Prompt_temple
import argparse
import json
import logging
from multimodal_dataset.graphics.chat_vis import visualize_conversation
from model.magi_llm import MAGI_llm
from model.score_compute import Score_compute
from model.magi_vision import MAGI_vision
from tools import get_dataset_config, print_config, set_credentials_config, set_seed
from multimodal_dataset.datasets.manage import DatasetManage
import os
import random
import numpy as np
import pickle as pkl
import torch


logger = logging.getLogger(__name__)

# Initialize the parser
parser = argparse.ArgumentParser(
    description="Visualize the PersonaChat dataset")

# Add arguments
############################################# Second Dataset setting #############################################
parser.add_argument("--config_data_yaml", type=str, default="/fsx/users/haboutal/home/github/MultimodalDataset/src/config/config_dataset.yaml",
                    help="path of yaml file for dataset configuration")
parser.add_argument("-n", "--name_second_dataset", type=str, default="PhotoChat",
                    help="name of the second dataset used for comparison")

############################################# Prompt setting #############################################
parser.add_argument("--prompt_type", type=str, default="COT_V5")

############################################# MAGI setting #############################################
parser.add_argument("--thresholds", nargs='+', type=float,
                    default=[0.21], help="thresholds of CLIP model")  # should be 0.2
parser.add_argument("--min_score_aesthetic", type=float,
                    default=0.51, help="thresholds of CLIP model")  # should be 0.51
parser.add_argument("--llm_model", type=str,
                    default="gpt-3.5-turbo", help="OpenAI model to use")
parser.add_argument("--diffusion_model", type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0", help="diffusion model to use. dreamstudio is another option")

############################################# Experiment setting #############################################
parser.add_argument("-o", "--output_path", type=str,
                    default="/fsx/users/haboutal/home/datasets/MAGI", help="path to save the results")
parser.add_argument("--credentials", type=str, default="/fsx/users/haboutal/home/github/MultimodalDataset/src/config/credentials.yaml",
                    help="path of yaml file for credentials of openai")
parser.add_argument("-s", "--seed", type=int,
                    default="8", help="seed for experiment")
parser.add_argument("--cuda_n", type=str,
                    default="0", help="cuda number for experiment")
parser.add_argument("--size_experiment", type=int, default=10)


# Parse arguments
args = parser.parse_args()
args.path_second_dataset, args.second_dataset_image_path = get_dataset_config(
    args.name_second_dataset, args.config_data_yaml)
args.output_path = os.path.join(args.output_path, args.name_second_dataset)
args.output_path = os.path.join(args.output_path, args.llm_model)
args.output_path = os.path.join(args.output_path, args.diffusion_model)
args.output_path = os.path.join(args.output_path, args.prompt_type)
args.path_meta_gpt = os.path.join(args.output_path, "mem_gpt.pkl")
args.image_path_magi = os.path.join(args.output_path, "images")

# Set gpu:
print(torch.cuda.current_device())

# Setting up API keys
set_credentials_config(args.credentials)

# Create the directory
try:
    os.makedirs(args.output_path)
except OSError:
    print("Creation of the directory %s failed" % args.output_path)
os.makedirs(args.image_path_magi, exist_ok=True)
# Setting up the logging
logging.basicConfig(level=logging.INFO, filemode='w', filename=os.path.join(
    args.output_path, "log.txt"))
logging.getLogger().addHandler(logging.StreamHandler())
print_config(args)

# set seed
set_seed(args.seed)

score_compute = Score_compute()
dataset_manager = DatasetManage(
    args.name_second_dataset, args.path_second_dataset, score_compute)
df = dataset_manager.df
candidate_indices = random.sample(range(len(df)), args.size_experiment)

# setting system prompt
prompt_template = Prompt_temple(args.prompt_type)
system_prompt = prompt_template.get_system_prompt()
logger.info("The system prompt is:\n {}".format(
    system_prompt.format(query="")))
prompt_dict = {}

num_dialogues = 0
num_gen_img_thresh = []
aesthetic_thresh_magi = []

# Setting up the LLM model
llm_model = MAGI_llm(llm_type=args.llm_model,
                     system_prompt=system_prompt, llm_answer_file=args.path_meta_gpt)
threshold = args.thresholds[0]

num_gen_images = 0
target_label = []
aesthetic_magi = []
magi_dataset = {}

for index in candidate_indices:
    new_prompt_log = {}
    torch.cuda.empty_cache()
    logger.info("Processing Dialogue ID: {}".format(str(index)))
    llm_content, clip_content, list_conversation = dataset_manager.prep_content(
        index)

    # creating a new folder for each experiment
    folder_path_results = os.path.join(os.path.join(
        args.output_path, str(threshold)), str(index))
    try:
        os.makedirs(folder_path_results)
    except OSError:
        print("Creation of the directory %s failed" % folder_path_results)

    # setting up the logging
    logger = logging.getLogger(f'experiment_{index}')
    logger.setLevel(logging.INFO)

    # Create a file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder_path_results, "log.txt"), 'w')
    fh.setLevel(logging.INFO)  # You can adjust the logging level as needed
    logger.addHandler(fh)
    logger.info(llm_content)
    prompts_diffusion_model, reason = llm_model.get_response(
        llm_content, use_llm_prediction=False, write_llm_prediction=True, logger=logger)

    # Generating Diffusion model images
    vision_model = MAGI_vision(diffusion_type=args.diffusion_model,
                               score_compute=score_compute, min_score_aesthetic=args.min_score_aesthetic)

    image_gen_magi = {}
    utterances_for_image_magi = set()
    for prompt in prompts_diffusion_model:
        index_utterance = int(prompt[0])
        prompt_diff = prompt[1]
        image, score_clip, score_aesthetic, path_save_image = vision_model.generate_image(
            prompt_diff, index_utterance, clip_content, args.image_path_magi, threshold=threshold, seed=args.seed, logger=logger)

        # image is None when the image is not generated
        if (image is None):
            new_prompt = llm_model.feedback(
                clip_content[index_utterance], prompt_diff)
            vision_model.reset_retry()
            image, score_clip, score_aesthetic, path_save_image = vision_model.generate_image(
                new_prompt, index_utterance, clip_content, args.image_path_magi, threshold=threshold, seed=args.seed, logger=logger)
            prompt_diff = new_prompt
            logger.info(
                "FEEDBACK LOOP USED with prompt: {}".format(prompt_diff))
            if (image is None):
                logger.info("EVEN FEEDBACK LOOP WAS NOT USEFUL")
                continue
        utterances_for_image_magi.add(index_utterance)
        logger.info("Utterance: {} | prompt: {} | score: {} | path image: {}".format(
            index_utterance, prompt_diff, score_clip, path_save_image))
        logger.info("reason: {}".format(reason))
        image_gen_magi[index_utterance] = path_save_image
        aesthetic_magi.append(score_aesthetic)
        new_prompt_log[index_utterance] = prompt_diff
    image_conv_index = set(image_gen_magi.keys())

    if len(image_conv_index) == 0:
        logger.info(
            "No images were generated for Dialogue ID: {}".format(str(index)))
        continue

    num_gen_images += len(image_conv_index)
    torch.cuda.empty_cache()

    # Visualize the conversation for model MAGID
    vis_conversation = []
    for i, element in enumerate(clip_content):
        vis_conversation.append(("txt", element))
        if (i in image_conv_index):
            vis_conversation.append(("img", image_gen_magi[i]))

    visualize_conversation(vis_conversation, os.path.join(
        folder_path_results, "vis_gen.png"))
    magi_dataset[index] = vis_conversation
    prompt_dict[index] = new_prompt_log
    num_dialogues += 1

    torch.cuda.empty_cache()

    # Evaluation number of generated images and aesthetic score
    logger.info(
        "####### Average Number of generated images: {}".format(num_gen_images))
    logger.info(
        "####### Number of generated dialogues: {}".format(num_dialogues))
    logger.info("####### Aesthetic score MAGI: {}".format(
        np.mean(np.array(aesthetic_magi))))

# saving the dataset
with open(os.path.join(args.output_path, "magi_dataset.json"), 'w') as fp:
    json.dump(magi_dataset, fp)
with open(os.path.join(args.output_path, "magi_dataset.pkl"), 'wb') as fp:
    pkl.dump(magi_dataset, fp)
with open(os.path.join(args.output_path, "magi_prompt.pkl"), 'wb') as fp:
    pkl.dump(prompt_dict, fp)
num_gen_img_thresh.append(num_gen_images)
aesthetic_thresh_magi.append(np.mean(np.array(aesthetic_magi)))


# logging the results
logger.info(" ####### Final results: ####### ")
logger.info("Results are saved in {}".format(args.output_path))
logger.info("####### Aesthetic score MAGI: {}".format(aesthetic_thresh_magi))
logger.info("####### Number of generated images MAGI: {}".format(
    num_gen_img_thresh))
logger.info("####### Number of generated dialogues: {}".format(num_dialogues))
with open(os.path.join(args.output_path, "results.pkl"), 'wb') as fp:
    fp.write(pkl.dumps({"num_gen_images": num_gen_img_thresh,
                        "aesthetic_score_magi": aesthetic_thresh_magi}))
