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
from collections import defaultdict
import re
from verl import DataProto
import random
# from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from functools import partial
from verl.utils.reward_score import prime_math


def sigmoid_k(x, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    if k == 0: # vanilla sigmoid
        return 1 / (1 + np.exp(-x))

    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))
    
    # Scale the output to [0, 1]
    return sigmoid

def threshold_t_sigmoid_k(x, t, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))

    result = 0 if sigmoid < t else sigmoid
    
    return result


def threshold_t_sigmoidv2_k(x, t, k=6):
    # concave curve
    if x < t:
        result = 0
    else:
        x = x - t
        x = x * k
        result = 1 / (1 + np.exp(-x))
    return result


def threshold_t_sigmoidv2fixed_k(x, t, k=6):
    if x < t:
        result = 0
    else:
        x = (x- t) * k
        result = 1 / (1 + np.exp(-x))  * ((1 - t) / 0.5) - ((1-t) - t) 
    
    return result


def threshold_t_sigmoidv3_k(x, t, k=6):
    # convex curve
    if x < t:
        result = 0
    else:
        x = (x - 1) * k
        result = 1 / (1 + np.exp(-x)) + 0.5
    return result


def leaky_relu_like(score, threshold, alpha=0.01):
    """
    Maps a score from [0, 1] to [0, 1] using a Leaky ReLU-like function.

    Parameters:
    - score: The input score in the range [0, 1].
    - threshold: The threshold below which the score is scaled.
    - alpha: The slope for scores below the threshold (default is 0.01).

    Returns:
    - The transformed score in the range [0, 1].
    """
    if score < threshold:
        return alpha * score
    else:
        return score


def threshold_t_tanh_k(score, t, k=6):
    # Apply tanh transformation with a configurable scaling factor
    transformed_score = (np.tanh(score * k - k / 2) + 1) / 2
    
    # Threshold values smaller than 0.05 to 0
    if transformed_score < t:
        transformed_score = 0
    
    return transformed_score


def format_reward(predict_str: str, format_mode='R1') -> float:
    def _validate_tags(input_string):
        if format_mode == 'R1':
            tags = ['<think>', '</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        else:
            raise ValueError(f"Unsupported format mode: {format_mode}")
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0

    if _validate_tags(predict_str) == 0.0:
        return 0.0
    if format_mode == 'R1':
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
    elif format_mode == 'R1_nothink':
        pattern = re.compile(r'.*<answer>.*</answer>.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)

    return 1.0 if match_result else 0.0


class ProbRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score_name=None, 
                 shaping_function_name=None, discrete_function_name=None, 
                 format_coefficient=0.1, save_results_dir=None, reward_type='pr',
                 gt_tokens_one_more=False, gt_tokens_one_more_adjusted=False, use = 0.9, explore = 0.1, use_scoreA = True, clip_actor = 10,
                 format_mode='R1') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # assert compute_score is None
        self.compute_score_name = compute_score_name
        print(f"{shaping_function_name=}")
        if shaping_function_name == 'identity':
            self.shaping_function = lambda x:x
        elif shaping_function_name == 'one_minus':
            self.shaping_function = lambda x: 1 - x
        elif shaping_function_name == 'random':
            self.shaping_function = lambda x: random.random()
        elif shaping_function_name.startswith('threshold'):
            threshold = float(shaping_function_name.split('_')[-1])
            self.shaping_function = lambda x: 0 if x < threshold else x
        elif shaping_function_name.startswith('sigmoid_'):
            print(f"Selecting sigmoid_k function.")
            k = float(shaping_function_name.split('_')[-1])
            self.shaping_function = partial(sigmoid_k, k=k)
        elif shaping_function_name.startswith('leaky_'):
            # e.g., leaky_0.05
            print(f"Using leaky-relu like function")
            threshold = float(shaping_function_name.split('_')[1])
            self.shaping_function = partial(leaky_relu_like, threshold=threshold)
        elif shaping_function_name.startswith('comp'): # compound
            # comp_threshold_0.3_sigmoid_6
            threshold = float(shaping_function_name.split('_')[2])
            k = float(shaping_function_name.split('_')[4])
            if 'sigmoidv2fixed' in shaping_function_name:
                print(f"Using sigmoid v2fixed")
                self.shaping_function = partial(threshold_t_sigmoidv2fixed_k, t=threshold, k=k)
            elif 'sigmoidv3' in shaping_function_name:
                print(f"Using sigmoid v3")
                self.shaping_function = partial(threshold_t_sigmoidv3_k, t=threshold, k=k)
            elif 'sigmoidv2' in shaping_function_name:
                print(f"Using sigmoid v2")
                self.shaping_function = partial(threshold_t_sigmoidv2_k, t=threshold, k=k)
            elif 'sigmoid' in shaping_function_name:
                print(f"Using sigmoid v1")
                self.shaping_function = partial(threshold_t_sigmoid_k, t=threshold, k=k)
            elif 'tanh' in shaping_function_name:
                self.shaping_function = partial(threshold_t_tanh_k, t=threshold, k=k)
            else:
                raise ValueError
        else:
            print(f"{shaping_function_name=}")
            raise NotImplementedError(f"{shaping_function_name=}")
        self.discrete_function_name = discrete_function_name
        self.format_coefficient = format_coefficient
        assert reward_type in ['pr', 'pr+vr']
        self.reward_type = reward_type
        self.format_mode = format_mode
        self.gt_tokens_one_more = gt_tokens_one_more
        self.gt_tokens_one_more_adjusted = gt_tokens_one_more_adjusted
        self.use = use
        self.explore = explore
        self.use_scoreA = use_scoreA
        self.clip_actor = clip_actor

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        extracted_answer_list = []
        # print(data)
        scoreA = self.compute_prob_scores(data, response_code="onlygroundtruth")
        scoreB = self.compute_prob_scores(data, response_code="groundtruth")
        scoreC = self.compute_prob_scores(data, response_code="original")
        # scoreD = np.abs(np.array(scoreB) - np.array(scoreC))
        # scoreD = np.max(np.array(scoreB) - np.array(scoreC), 0) * 
        # scoreD = np.array(scoreC) - np.array(scoreB)

        scoreE = [data_item.non_tensor_batch['reward_model'].get('scoreA', 0.0) for data_item in data]
        
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreA_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreB_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreC_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreD_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreE_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            response_ids = data_item.batch['responses'] # len(response_ids): 1024
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # 329
            valid_response_ids = response_ids[:valid_response_length] 

            extracted_answer = self.extracted_answer_valid(valid_response_ids)

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences) # <|im_start|>system\n..the answer is: \boxed{10, 30, 40}<|im_end|>

            predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>
            format_score = format_reward(predict_str=predict_str, format_mode=self.format_mode)
            if self.clip_actor < 1 :
                if scoreC[i] > scoreB[i]:
                    score_e = scoreC[i] - scoreB[i]
                else:
                    score_e = ((scoreB[i] - scoreC[i]) <= self.clip_actor)*(scoreB[i] - scoreC[i])
            else:

                if scoreC[i] > scoreB[i]:
                    score_e = scoreC[i] - scoreB[i]
                    # score_e = 0
                else:
                    score_e = ((scoreB[i] - scoreC[i]) <= scoreA[i] / self.clip_actor)*(scoreB[i] - scoreC[i])

            if self.use_scoreA == True:
                score_delta = self.use * (scoreA[i] - scoreE[i]) + self.explore * score_e
            else:
                score_delta = self.use * scoreA[i] + self.explore * score_e

            score_delta = self.shaping_function(score_delta)
            if self.discrete_function_name is not None and self.discrete_function_name != 'identity':
                if self.discrete_function_name.startswith('bin_'):
                    num_bins = int(self.discrete_function_name.split('_')[1])
                    score_delta = self.map_to_bins(score_delta, num_bins)

            if self.format_coefficient == -1:
                score = score_delta if format_score == 1 else -1
            else:
                score = (1 - self.format_coefficient) * (score_delta) + self.format_coefficient * format_score
            reward_tensor[i, valid_response_length - 1] = score
            scoreA_tensor[i, valid_response_length - 1] = scoreA[i]
            scoreB_tensor[i, valid_response_length - 1] = scoreB[i]
            scoreC_tensor[i, valid_response_length - 1] = scoreC[i]
            scoreD_tensor[i, valid_response_length - 1] = score_e
            scoreE_tensor[i, valid_response_length - 1] = score_e
            format_reward_tensor[i, valid_response_length - 1] = format_score

            data_source = data_item.non_tensor_batch['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"{data_source=} {sequences_str=}", flush=True)

            extracted_answer_list.append(extracted_answer)

        return reward_tensor, scoreD_tensor, scoreA_tensor, scoreC_tensor, scoreB_tensor, scoreE_tensor, format_reward_tensor, extracted_answer_list


    @staticmethod
    def map_to_bins(score: float, num_bins: int) -> float:
        """
        Maps a score in [0,1] to the nearest discrete value based on num_bins.
        
        :param score: A float between 0 and 1.
        :param num_bins: The number of discrete bins.
        :return: The mapped discrete value.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        
        step = 1 / num_bins
        return round(score / step) * step

    def compute_scoreB(self, old_log_probs, ground_truth_mask):
        if ground_truth_mask.sum() == 0:
            scoreB = 0
        else:
            old_log_probs_in_gt = old_log_probs[ground_truth_mask.bool()]
            if self.gt_tokens_one_more and self.gt_tokens_one_more_adjusted:
                old_log_probs_in_gt[-1] = min(old_log_probs_in_gt[-1], np.log(0.5))
            # Convert to BF32 for higher precision calculations
            old_log_probs_in_gt = old_log_probs_in_gt.to(torch.float32)
            # mean of probs
            if self.compute_score_name == 'mean_exp_log_softmax':
                scoreB = torch.mean(torch.exp(old_log_probs_in_gt)).item()
            # mean log probs
            elif self.compute_score_name == 'mean_log_softmax':
                scoreB = torch.mean(old_log_probs_in_gt).item()
            # product of probs
            elif self.compute_score_name == 'exp_sum_log_softmax':
                scoreB = torch.exp(torch.sum(old_log_probs_in_gt)).item()
            # root of the product of probs
            elif self.compute_score_name == 'exp_mean_log_softmax':
                scoreB = torch.exp(torch.mean(old_log_probs_in_gt)).item() 
            else:
                raise ValueError
            # Convert the final score back to BF16 if necessary
            scoreB = torch.tensor(scoreB, dtype=torch.bfloat16).item()

        return scoreB

    def compute_scoreA_scoreB_and_extracted_answer(self, data_item, valid_response_ids):
        data_source = data_item.non_tensor_batch.get("data_source", "WebInstruct-verified")
        if self.reward_type == 'pr+vr' and (any(dataset_name in data_source for dataset_name in ["numina_cn_k12", "numina_synthetic_math", "numina_olympiads", 
                                                                "numina_synthetic_amc", "numina_aops_forum", "numina_amc_aime",
                                                                "Math-500", "AIME2024", "AMC2023", "DAPO-Math-17k",
                                                                "OlympiadBench", "Minerva", "simplelr_deepscaler"])):

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
            res = prime_math.compute_score(solution_str, ground_truth)
            scoreB = float(res[0])
            scoreA = 0
            extracted_answer = res[1]
        else:
            scoreB = self.compute_scoreB(data_item.batch['old_log_probs_pr'], data_item.batch['ground_truth_mask_pr']) # shape: [1024]
            # scoreA = data_item.non_tensor_batch['reward_model'].get('scoreA', 0.0)

            predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>
            match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
            extracted_answer = match.group(1).strip() if match else ""
        return scoreB, extracted_answer

    def compute_prob_scores_item(self, data, response_code="onlygroundtruth"):
        if response_code == "original":
            log_probs = data.batch["old_log_probs"]
            response_mask = data.batch["response_mask_pr"]
        elif response_code == "groundtruth":
            log_probs = data.batch["old_log_probs_pr"]
            response_mask = data.batch["ground_truth_response_mask_pr"]
        elif response_code == "onlygroundtruth":
            log_probs = data.batch["old_log_probs_pr"]
            response_mask = data.batch["ground_truth_mask_pr"]
        else:
            raise ValueError("error response code for computing pr scores!!!")
        # print(log_probs,response_mask)
        return self.compute_scoreB(log_probs, response_mask)
    
    def compute_prob_scores(self, data, response_code="onlygroundtruth"):
        score = [self.compute_prob_scores_item(data[i], response_code) for i in range(len(data))]
        return score

    def extracted_answer_valid(self, valid_response_ids):
        predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>
        match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
        extracted_answer = match.group(1).strip() if match else ""
        return extracted_answer