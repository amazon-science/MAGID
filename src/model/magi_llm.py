# This class is responsible for the MAGI pipeline (Multimodal Data Set Craetion from Text-only Data Set)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pickle as pkl
import openai
import fcntl
import re
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from model.sage_maker_llm import SageMakerLLM


class MAGI_llm:
    def __init__(self, llm_type, system_prompt, llm_answer_file) -> None:
        self.llm_type = llm_type
        self.counter_write = 1
        self.sagemaker_endpoint = None
        self.llm_answer_file = llm_answer_file  # file containing the saved llm answer
        self.system_prompt = system_prompt
        self.chat = self.construct_llm_pipeline()
        try:
            with open(self.llm_answer_file, "rb") as file:
                fcntl.flock(file, fcntl.LOCK_EX)  # Exclusive lock, blocking
                self.llm_predicts_dict = pkl.load(file)
                fcntl.flock(file, fcntl.LOCK_UN)  # Unlock
                assert isinstance(self.llm_predicts_dict, dict)
        except:
            self.llm_predicts_dict = {}

    def construct_llm_pipeline(self):
        if self.llm_type == "gpt-3.5-turbo":
            return OpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
        elif self.llm_type == "gpt-4":
            return OpenAI(temperature=0.3, model_name="gpt-4")
        else:
            raise Exception("LLM type not supported")

    def get_response(self, query, use_llm_prediction=True, write_llm_prediction=True, logger=None):
        prompt = self.system_prompt.format(query=query)
        # print(prompt)
        key_mem = str(prompt)[-300:]
        if use_llm_prediction:
            if tuple(key_mem) in self.llm_predicts_dict.keys():
                response, reason_prompt = self.llm_predicts_dict[tuple(
                    key_mem)]
                print("Conversation already annotated")
                return response, reason_prompt
        # print("prompt: ", prompt )
        llm_response = self.dispatch_llm(prompt)
        logger.info("######## LLM Response: ########")
        logger.info(llm_response)
        logger.info("######## END OF LLM Response: ########")
        response, reason_prompt = self.polish_response(llm_response, query)
        if write_llm_prediction:
            self.llm_predicts_dict[tuple(key_mem)] = (response, reason_prompt)
            if(self.counter_write % 10 == 0):
                try:
                    with open(self.llm_answer_file, "rb") as file:
                        # Exclusive lock, blocking
                        fcntl.flock(file, fcntl.LOCK_EX)
                        old_llm_predicts_dict = pkl.load(file)
                        fcntl.flock(file, fcntl.LOCK_UN)  # Unlock
                        assert isinstance(self.llm_predicts_dict, dict)
                except:
                    old_llm_predicts_dict = {}
                with open(self.llm_answer_file, "wb") as file:
                    # Exclusive lock, blocking
                    fcntl.flock(file, fcntl.LOCK_EX)
                    self.llm_predicts_dict = {
                        **old_llm_predicts_dict, **self.llm_predicts_dict}
                    pkl.dump(self.llm_predicts_dict, file)
                    fcntl.flock(file, fcntl.LOCK_UN)  # Unlock
        self.counter_write += 1
        return response, reason_prompt

    def dispatch_llm(self, prompt):
        if self.llm_type == "gpt-4" or self.llm_type == "gpt-3.5-turbo":
            return self.chat(prompt)
        else:
            return self.sagemaker_endpoint.main(prompt)

    def polish_response(self, response, query):
        assert type(response) == str
        pattern = re.compile(r"\bno\s*image\b", re.IGNORECASE)
        prompts = []
        if self.llm_type != "gpt-4" and self.llm_type != "gpt-3.5-turbo":
            response = response.split("AI Answer")[-1]
        if "<reason>" not in response or "</reason>" not in response:
            reason_prompt = ""
        else:
            reason_prompt = re.findall(
                r"<reason>(.*?)</reason>", response, re.DOTALL)[0]
        if "<result>" not in response or "</result>" not in response:
            return prompts, reason_prompt
        filtered_answer = re.findall(
            r"<result>(.*?)</result>", response, re.DOTALL)[0]
        if "\\n" in filtered_answer or "\\\n" in filtered_answer or "\\\\n" in filtered_answer:
            filtered_answer = filtered_answer.replace("\\n", "\n")
            filtered_answer = filtered_answer.replace("\\\n", "\n")
            filtered_answer = filtered_answer.replace("\\\\n", "\n")
        if abs(len(filtered_answer) - len(query)) <= 10:
            return prompts, reason_prompt
        if "\n" not in filtered_answer:
            for i, element in enumerate(filtered_answer.split("Utterance")):
                if (
                    len(element) <= 5
                    or re.search(r"\d+", element) is None
                    or bool(pattern.search(element))
                ):
                    continue
                int_value = re.search(r"\d+", element).group()
                last_int_index = re.search(r"\d+", element).end() + 1
                prompts.append([int_value, element[last_int_index:]])
        else:
            for i, element in enumerate(filtered_answer.split("\n")):
                if (
                    len(element) <= 20
                    or re.search(r"\d+", element) is None
                    or bool(pattern.search(element))
                ):
                    continue
                int_value = re.search(r"\d+", element).group()
                last_int_index = re.search(r"\d+", element).end() + 1
                prompts.append([int_value, element[last_int_index:]])
        return prompts, reason_prompt

    def feedback(self, content, previous_answer):
        answer = self.get_llm_feedback(content, previous_answer)
        return answer

    def get_llm_feedback(self, content, previous_answer):
        if self.llm_type == "gpt-4" or self.llm_type == "gpt-3.5-turbo":
            completion = openai.ChatCompletion.create(
                model=self.llm_type,
                messages=[
                    {
                        "role": "system",
                        "content": """Your task is for a given sentence, provide the image description that if included with the sentence can support the sentence and make it more engaging. """
                        + """ The previous answer that was not accepted was: ' """
                        + previous_answer
                        + """ ' For the following sentence, Provide a shorter image description that better supports the conversation: """,
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0,
            )
            answer = completion["choices"][0]["message"]["content"]
            return answer
        else:
            raise Exception("LLM type not supported")
