import multiprocessing as mp
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from copy import deepcopy
import re
import argparse
import datasets

# def load_data(name, split):
#     data = datasets.load_dataset(name)
#     return data[split][sentence_name], data[split][label_name]

class remote_LLM:
    def __init__(self, model_url, model_name):
        self.model_url = model_url
        self.model_name = model_name
        
    def run_inference(self, description, sentence, label, prompt, temp=0):
        """
        Simulated execution of a prompt with a given temperature.
        Replace the body with your actual model inference logic.
        """
        print(f"[{self.model_name}] Started meta-prompt generation with prompt='', temp={temp}")
        
        print("okk")
        model_prompt = prompt.format(description=description, sentence=sentence, label=label)
        # print(model_prompt_tmpl)
        requests_res = requests.post('http://10.244.50.59:1234/model', json={'prompt': model_prompt}).json()
        clean_result_str = requests_res["output"]
        
        
        # result = f"[{self.model_name}] Result for prompt='' at temp={temp}"
        print(f"[{self.model_name}] Finished execution.")
        return clean_result_str

    def summarize_patterns(self, sentence, label, prompt, temp=0):
        """
        Simulated execution of a prompt with a given temperature.
        Replace the body with your actual model inference logic.
        """
        print(f"[{self.model_name}] Starting summary with prompt='', temp={temp}")
        
        print("okk")
        model_prompt = prompt.format(sentence=sentence, label=label)
        # print(model_prompt_tmpl)
        requests_res = requests.post('http://10.244.50.59:1234/model', json={'prompt': model_prompt}).json()
        clean_result_str = requests_res["output"]
        
        
        # result = f"[{self.model_name}] Result for prompt='' at temp={temp}"
        print(f"[{self.model_name}] Finished execution.")
        return clean_result_str

    def verify_patterns(self, sentence, label, prediction, prompt, temp=0):
        """
        Simulated execution of a prompt with a given temperature.
        Replace the body with your actual model inference logic.
        """
        print(f"[{self.model_name}] Starting verification with prompt='', temp={temp}")
        
        print("okk")
        model_prompt = prompt.format(sentence=sentence, label=label, model_predict=prediction)
        # print(model_prompt_tmpl)
        requests_res = requests.post('http://10.244.50.59:1234/model', json={'prompt': model_prompt}).json()
        clean_result_str = requests_res["output"]
        
        
        # result = f"[{self.model_name}] Result for prompt='' at temp={temp}"
        print(f"[{self.model_name}] Finished execution.")
        return clean_result_str

    def pattern_work(self, sentence, prompt, all_patterns, temp=0):
        """
        Pattern Selection and extraction using selected patterns.
        """
        print(f"[{self.model_name}] Starting selection/extraction with prompt='', temp={temp}")
        
        print("okk")
        model_prompt = prompt.format(sentence=sentence, patterns=all_patterns)
        # print(model_prompt_tmpl)
        requests_res = requests.post('http://10.244.50.59:1234/model', json={'prompt': model_prompt}).json()
        clean_result_str = requests_res["output"]
        
        
        # result = f"[{self.model_name}] Result for prompt='' at temp={temp}"
        print(f"[{self.model_name}] Finished execution.")
        return clean_result_str

    def predict_single(self, sentence, prompt, temp=0):
        """
        Pattern Selection and extraction using selected patterns.
        """
        print(f"[{self.model_name}] Starting baseline with prompt='', temp={temp}")
        
        print("okk")
        model_prompt = prompt.format(sentence=sentence)

        # print(model_prompt_tmpl)
        requests_res = requests.post('http://10.244.50.59:1234/model', json={'prompt': model_prompt}).json()
        clean_result_str = requests_res["output"]
        # print(clean_result_str)
        # print(f"[{self.model_name}] Finished execution.")
        try:
            res = re.search(r'<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>', clean_result_str, re.I|re.S).group(1)
            # print(res)
            return res
        except Exception as e:
            print(e)
            return clean_result_str
            
        
        
        # result = f"[{self.model_name}] Result for prompt='' at temp={temp}"
        
    


    

        
# 2. turn single-quotes into double-quotes
        # match = re.search(r"<oj>(.*?)</oj>", clean_result_str, re.DOTALL)
        # if match:
        #     json_str = match.group(1)
        # filtered_json = re.search(r'<oj>(.*?)</oj>', clean_result_str, re.DOTALL).group(1)

class local_LLM():
    def __init__(self, model_name, args):
        self.model_name = model_name

    def run_inference(self):
        return