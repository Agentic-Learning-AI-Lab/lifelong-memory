class PromptTemplate:
    """ Prompt template for (M)LLMs that take textual inputs, such as GPT3.5, GPT4, Vicuna. """

    sep = "\n\n###\n\n"

    def __init__(self, system_prompt, user_begin_text, user_end_text):
        self.system_prompt = system_prompt
        self.user_begin_text = user_begin_text
        self.user_end_text = user_end_text

    def get_user_prompt(self, queries, captions):
        return self.user_begin_text + self.sep + "Questions:" + self.sep + queries + self.sep + "Memory:" + self.sep + captions + self.sep + self.user_end_text
    
    def get_system_prompt(self):
        return self.system_prompt
    

class VisionPromptTemplate(PromptTemplate):
    """ Prompt template for LLMs that take image inputs, such as GPT4V.  """

    def get_user_prompt(self, queries, images):
        return [self.user_begin_text + self.user_end_text + self.sep + "Questions:" + self.sep + queries, *images]

def load_template(name):
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
    if 'vision' in name:
        user_begin = "Given frames of the egocentric videos, employ your advanced memory recall capabilities to pinpoint the time intervals that best respond to the questions provided. Ensure each answer encompasses not just the event in question but also the relevant context before and after. If the details are too vague or insufficient for a confident recall, indicate 'NA'. "
        user_end = "Please provide a TSV with columns: query_index, predictions, explanation, confidence, where predictions are the indices of the frames."
        return VisionPromptTemplate(system_prompt, user_begin, user_end)
    if 'simple' in name:
        system_prompt = "You are individual C, with others represented as O. You want to recall your memories to locate the timestamps that can answer the given queries."
        user_begin = "As individual C, with others as O, employ your advanced memory recall capabilities to pinpoint the timestamps that best respond to the questions provided. If the details are too vague or insufficient for a confident recall, indicate 'NA'."
        user_end = "Please provide a TSV with columns: query_index, predictions."
    return PromptTemplate(system_prompt, user_begin, user_end)