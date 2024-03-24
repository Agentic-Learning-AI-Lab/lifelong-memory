class PromptTemplate:
    """ Prompt template for (M)LLMs that take textual inputs, such as GPT3.5, GPT4, Vicuna. """
    def __init__(self, system_prompt, user_begin_text, user_end_text):
        self.system_prompt = system_prompt
        self.user_begin_text = user_begin_text
        self.user_end_text = user_end_text
        self.sep = "\n\n###\n\n"
    
    def get_system_prompt(self):
        return self.system_prompt
    
class NLQPromptTemplate(PromptTemplate):
    """ Prompt template for NLQ. """
    def __init__(self, system_prompt, user_begin_text, user_end_text):
        super().__init__(system_prompt, user_begin_text, user_end_text)
        self.search_key = ["cid", "vid"]
        self.output_format = "tsv"

    def get_user_prompt(self, queries, captions):
        print(self.user_begin_text + self.sep + "Questions:" + self.sep + queries + self.sep + "Memory:" + self.sep + captions + self.sep + self.user_end_text)
        return self.user_begin_text + self.sep + "Questions:" + self.sep + queries + self.sep + "Memory:" + self.sep + captions + self.sep + self.user_end_text
    
class QAPromptTemplate(PromptTemplate):
    """ Prompt template for QA. """
    def __init__(self, system_prompt, user_begin_text, user_end_text):
        super().__init__(system_prompt, user_begin_text, user_end_text)
        self.search_key = ["q_uid"]
        self.output_format = "dic"

    def get_user_prompt(self, queries, captions):
        return "Memory:" + self.sep + captions + self.sep + queries + self.sep + self.user_end_text
    
class VisionNLQPromptTemplate(NLQPromptTemplate):
    """ NLQ prompt template for LLMs that take image inputs, such as GPT4V.  """
    def get_user_prompt(self, queries, images):
        return [self.user_begin_text + self.user_end_text + self.sep + "Questions:" + self.sep + queries, *images]

def load_template(name):
    name = name.lower()
    if 'nlq' in name:
        system_prompt = "You are individual C, with others represented as O. In your responses to questions about past events, it is vital to provide not only the key moment but also the relevant context. To enhance the clarity and reliability of your answers, please also indicate your confidence level in each response, with 1 being the lowest and 3 being the highest. Follow these guidelines: \n" + \
                        "1. Incorporate Context: Expand your answers to include not just the central event but also the context preceding and following it. \n " + \
                        "2. Unify Related Actions: When a question requires a sequence of actions, such as 'Where did I put the scarf after I closed the door?', merge all relevant events into a single interval that conveys the full story. \n " + \
                        "3. Opt for Broad Understanding: Favor comprehensive intervals that cover all relevant details over more precise but less informative ones. If the information is too vague, respond with 'NA' and include your confidence level to reflect the certainty of your response. \n " + \
                        "4. Assign a Confidence Level: After providing a time interval, add a confidence level to each response: " + \
                            "- Level 1: The information is present, but the context is not clear or the captions are ambiguous. " + \
                            "- Level 2: The information is fairly clear and context is somewhat discernible, but there is still some uncertainty." + \
                            "- Level 3: The information and context are clear and well-supported by the captions, ensuring a high level of confidence."
        user_begin = "As individual C, with others as O, employ your advanced memory recall capabilities to pinpoint the timestamps that best respond to the questions provided. Ensure each answer encompasses not just the event in question but also the relevant context before and after. If the details are too vague or insufficient for a confident recall, indicate 'NA'."
        user_end = "Please provide a TSV with columns: query_index, predictions, explanation, confidence."
        return NLQPromptTemplate(system_prompt, user_begin, user_end)
    if 'qa' in name:
        system_prompt = "You are presented with a textual description of a video clip. Your task is to answer a question related to this video, choosing the correct option out of five possible answers. " + \
                        "It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 3, where 1 indicates low confidence and 3 signifies high confidence. " + \
                        "Please provide a concise one-sentence explanation for your chosen answer. If you are uncertain about the correct option, select the one that seems closest to being correct. "           
        user_end = "The dictionary with keys of prediction, explanation, confidence, where prediction is a number. "
        return QAPromptTemplate(system_prompt, "", user_end)
    raise ValueError("Invalid task name. Expected to contain NLQ or QA.")