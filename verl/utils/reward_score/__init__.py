# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

import re
import os
try:
    from .client import ChatClient, OpenAIClient
    from . import prime_math
    from . import prime_math_train
    from . import multi_choice
except:
    from client import ChatClient, OpenAIClient
    import prime_math
    import prime_math_train
    import multi_choice

used_model = os.environ.get('USED_MODEL', None) # 'gpt'
if used_model in ['gpt-4o', 'gpt-4.1']:
    client = OpenAIClient(
        api_key=os.environ.get('OPENAI_API_KEY', None),
        api_base=os.environ.get('OPENAI_API_BASE', None),
        model=used_model,
        max_tokens=12288,
    )
elif used_model == 'no_api':
    print("We do not use API for evaluation")
else:
    client_ip = os.environ.get('CLIENT_IP', 'http://127.0.0.1:8001')
    print(client_ip)
    client = ChatClient(server_url=client_ip, model="qwen")


prompt = '''**Task:**  
Given a **question** and a **model_response** to a multiple-choice problem, extract the exact option (e.g., A, B, C, D) that the model selects as its answer. If the response does not explicitly state an option but implies an answer, infer the most likely choice. If no clear option is provided, return "Unclear."

### Example 1:  
- **Question**: ```What is the capital of France? (A) London (B) Paris (C) Berlin (D) Rome```  
- **Model Response**: ```The correct answer is Paris, which corresponds to option B.```  
- **Extracted Option**: \\boxed{{B}}

### Example 2:  
- **Question**: ```Which element has the chemical symbol 'O'? (A) Gold (B) Oxygen (C) Carbon (D) Iron```  
- **Model Response**: ```Oxygen is the element with symbol 'O'.```  
- **Extracted Option**: \\boxed{{B}} 

### Your Task:  
- **Question**: ```{question}```
- **Model Response**: ```{model_response}```
NOTE: Please do not try to solve the problem or provide an answer! Instead, focus on extracting the option!
Please think step-by-step and put your final extracted option, i.e., A, B, C, etc in \\boxed{{}}.'''


math_prompt = '''**Task:**  
Given a **question** and a **model_response** to a math problem, extract the concise answer from the **model_response**.

Example 1. **Math (Symbolic):**  
- Model Answer: ```x = 2```  
- Ground Truth: ```2```  
- Output: \\boxed{{Y}}  

Example 2. **Math (Unit Conversion):**  
- Problem: ```Convert 300 seconds to minutes.```  
- Model Answer: ```5 minutes.```  
- Ground Truth: ```5```  
- Output: \\boxed{{Y}}


Example 2. **Math:**  
- Model Answer: ```x = \\frac{{1}}{{3}}```  
- Ground Truth: ```x = \\frac{{2}}{{3}}```  
- Output: \\boxed{{N}}

### Your Task:  
- **Question**: ```{question}```
- **Model Response**: ```{model_response}```
- **Ground Truth**: ```{ground_truth}```

NOTE: Please do NOT try to solve the problem or provide an answer by yourself! Instead, focus on EXTRACT the origin answer from the **model_response** and then decide whether this answer is aligned with the ground truth (yes for aligned, no for not-aligned).
Please put your final extracted option, i.e., "Y" or "N" in \\boxed{{}}.'''




def format_reward(predict_str: str, prompt_str: str, format_mode='R1') -> float:
    def _validate_tags(input_string):
        if format_mode == 'R1':
            tags = ['<think>', '</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0

    if '<answer>' in prompt_str and '</answer>' in prompt_str:
        if _validate_tags(predict_str) == 0.0:
            return 0.0
        if format_mode == 'R1':
            pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
        elif format_mode == 'R1_nothink':
            pattern = re.compile(r'.*<answer>.*</answer>.*', re.DOTALL)
        match_result = re.fullmatch(pattern, predict_str)
    elif '\\boxed{' in prompt_str:
        pattern = re.compile(r'.*\\boxed\{.*\}.*', re.DOTALL)
        match_result = re.fullmatch(pattern, predict_str)
    else:
        return 1.0
    
    return 1.0 if match_result else 0.0

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None, phase='train', format_mode='R1'):
    format_score = format_reward(solution_str, prompt_str, format_mode=format_mode)
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif 'gpqa_diamond' in data_source or 'HellaSwag' in data_source or 'WebInstruct-verified-val' in data_source or 'MMLUPro' in data_source or 'SuperGPQA' in data_source:
        
        if 'MMLUPro' in data_source or 'SuperGPQA' in data_source:
            options = 'ABCDEFGHIJKLMNOP'
        elif 'gpqa_diamond' in data_source or 'HellaSwag' in data_source or 'WebInstruct-verified-val' in data_source:
            options = 'ABCD'
        else:
            raise ValueError

        res = multi_choice.compute_score(solution_str, ground_truth, options=options, prompt_str=prompt_str)
 
 
    elif any(dataset_name in data_source for dataset_name in ["numina_cn_k12", "numina_synthetic_math", "numina_olympiads", 
                                                              "numina_synthetic_amc", "numina_aops_forum", "numina_amc_aime",
                                                              "Math-500", "AIME2024", "AMC2023", "DAPO-Math-17k", 
                                                              "WebInstruct-verified", "orz_math_57k_collection", "MATH",
                                                              'OlympiadBench', "Minerva", "TheoremQA", "simplelr_deepscaler",
                                                              "allenai-tulu-3-sft-mixture-train-tulu_v3.9_wildchat_100k", "allenai-tulu-3-sft-mixture-train-personahub_ifdata_manual_seed_v3_29980",
                                                              "allenai-tulu-3-sft-mixture-train-flan_v2_converted", "allenai-tulu-3-sft-mixture-train-no_robots_converted",
                                                              "allenai-tulu-3-sft-mixture-train-tulu_v3.9_sciriff_10k",
                                                              "allenai-tulu-3-sft-mixture-train-tulu_v3.9_table_gpt_5k",
                                                              "allenai-tulu-3-sft-mixture-train-oasst1_converted",
                                                              "simplelr_deepscaler"]):
        if phase == 'train':
            res = prime_math_train.compute_score(solution_str, ground_truth)
        elif phase == 'validation':
            res = prime_math.compute_score(solution_str, ground_truth)
        else:
            raise NotImplementedError(f"phase {phase} not in the implementation")
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)

    else:
        print(f"{data_source=} not in the implementation")
        raise NotImplementedError

    from_judge = 'rule'
    judge_response = ''
    if used_model in ['gpt-4o', 'gpt-4.1']:
        flag = True
    elif used_model == 'no_api':
        flag = False
    else:
        flag = (res[0] != 1 or res[0] != True)
 
    if phase == 'validation' and flag:
        question = get_raw_question_from_prompt(prompt_str)
        model_response = solution_str
        if 'user\n' in model_response:
            model_response = model_response.split('user\n')[0]  # # avoid the issue that base model appends additional conversation
        if any(dataset_name in data_source for dataset_name in ["Math-500", "AMC2023", "OlympiadBench", "Minerva", "TheoremQA", "AIME2024"]):
            selected_prompt = math_prompt
        elif 'gpqa_diamond' in data_source or 'WebInstruct-verified-val' in data_source or 'MMLUPro' in data_source or 'SuperGPQA' in data_source:
            selected_prompt = prompt
        else:
            raise NotImplementedError(f"{data_source=} not in the implementation")
        sent_prompt = selected_prompt.format(question=question, ground_truth=ground_truth, model_response=solution_str)
        if used_model in ['gpt-4o', 'gpt-4.1']:
            try:
                response = client.chat_sync_retry(
                    user_prompt=sent_prompt,
                    model=used_model,
                )
            except Exception as e:
                response = 'Error'
        else:
            response = client.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": sent_prompt}
                ],
                max_tokens=12288,
            )
        if response:
            if used_model in ['gpt-4o', 'gpt-4.1']:
                judge_response = response
            else:
                judge_response = response["choices"][0]["message"]["content"]
            extracted_option = extract_option(judge_response.lower())
            if extracted_option:
                extracted_option = extracted_option.strip().upper()
            correct = extracted_option and ((selected_prompt == prompt and extracted_option.lower() == str(ground_truth).lower()) or (selected_prompt == math_prompt and extracted_option == 'Y'))
            if correct:
                res = (1.0 if isinstance(res[0], (int, float)) else True, res[1])
                from_judge = 'model'
        

    return float(res[0]), format_score, res[1], from_judge, judge_response




def get_raw_question_from_prompt(prompt_str):
    if 'user\n' in prompt_str:
        question = 'user\n'.join(prompt_str.split('user\n')[1:])
    elif 'User: ' in prompt_str:
        question = 'User: '.join(prompt_str.split('User: ')[1:])
    else:
        raise ValueError(f"Invalid prompt format: {prompt_str}")
    if 'Question:\n' in question: # MMLUPro
        question = question.split('Question:\n')[1]
    
    question = question.replace('Format your response as follows: "The correct answer is (insert answer here)"', '') # WebInstruct-verified-OC, gpqa_diamond-OC
    question = question.replace(r'Please reason step by step, and put your final answer within \boxed{}.', '') 
    question = question.strip()
    if 'What is the correct answer to this question: ' in question:
        question = question.replace('What is the correct answer to this question: ', '')
    if question.endswith('\nassistant\n'):
        question = question[:-len('\nassistant\n')]
    while '<|im_end|>' in question:
        question = question.replace('<|im_end|>', '')
    while '<|im_start|>' in question:
        question = question.replace('<|im_start|>', '')
    return question.strip()

def extract_option(judge_response):
    # Pattern to match \boxed{<valid option>}, where valid option is A-P
    pattern = r'\\boxed\{([a-z])\}'
    match = re.findall(pattern, judge_response)
    if match:
        return match[-1]  # Return the extracted option (A-P)
    return None  # Return None if no valid option is found