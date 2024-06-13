from langchain import FewShotPromptTemplate, PromptTemplate
import yaml


class Prompt_temple():

    def __init__(self, prompt_type) -> None:
        self.exampler_id, self.prefix_id, self.suffix_id = self.get_prompt_type(
            prompt_type)
        with open('/fsx/users/haboutal/home/github/MultimodalDataset/src/config/prompt.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def get_prompt_type(self, prompt_type):
        if prompt_type == "COT_V2":
            return "COT_V2", "V1", "V1"
        elif prompt_type == "COT_V1":
            return "COT_V1", "V2", "V1"
        elif prompt_type == "Zero_V1":
            return "V0", "V3", "V1"
        elif prompt_type == "COT_V4":  # For MMDialouge
            return "COT_V4", "V3", "V1"
        elif prompt_type == "COT_V6":  # For few shot mmdialog
            return "COT_V6", "V3", "V1"
        elif prompt_type == "COT_V5":  # For PhotoChat
            return "COT_V5", "V4", "V1"
        else:
            raise Exception("Prompt type not found")

    def get_conversation_prompt(self):
        exampler = self.get_exampler()
        example_template = """
        User: \n{query}
        AI Answer: \n{answer}
        """
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )
        few_shot_prompt_template = FewShotPromptTemplate(
            examples=exampler,
            example_prompt=example_prompt,
            prefix="{query}",
            suffix="",
            input_variables=["query"],
            example_separator="\n"
        )
        return few_shot_prompt_template

    def get_system_prompt(self):
        exampler = self.get_exampler()
        prefix = self.get_prefix()
        suffix = self.get_suffix()

        # create a example template
        example_template = """
        User: \n{query}
        AI Answer: \n{answer}
        """

        # create a prompt example from above template
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )

        # For the case where we use Chain of Thought for Few Shot learning
        if (exampler is not None):
            few_shot_prompt_template = FewShotPromptTemplate(
                examples=exampler,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["query"],
                example_separator="\n\n"
            )
            output_prompt = few_shot_prompt_template
        else:

            # for the case where we do not use Chain of Thought for Zero Shot learning
            output_prompt = prefix + suffix
        return output_prompt

    def get_exampler(self):
        if self.exampler_id == "COT_V1":
            return self.config["exampler"]["V1"]
        elif self.exampler_id == "COT_V2":
            return self.config["exampler"]["V2"]
        elif self.exampler_id == "COT_V3":
            return self.config["exampler"]["V3"]
        elif self.exampler_id == "COT_V4":
            return self.config["exampler"]["V4"]
        elif self.exampler_id == "COT_V5":
            return self.config["exampler"]["V5"]
        elif self.exampler_id == "COT_V6":
            return self.config["exampler"]["V6"]
        elif self.exampler_id == "V0":
            return None
        else:
            raise ValueError("exampler_id is not valid")

    def get_prefix(self):
        if self.prefix_id == "V1":
            return self.config["prefix"]["V1"]
        elif self.prefix_id == "V2":
            return self.config["prefix"]["V2"]
        elif self.prefix_id == "V3":
            return self.config["prefix"]["V3"]
        elif self.prefix_id == "V4":
            return self.config["prefix"]["V4"]
        else:
            raise ValueError("prefix_id is not valid")

    def get_suffix(self):
        return self.suffix_v1()

    def suffix_v1(self):
        suffix = """
        User:\n {query}
        \n        AI Answer:\n """
        return suffix
