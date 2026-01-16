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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import os
from datetime import datetime
import pandas as pd
import torch


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_results_dir=None, phase='train', format_mode='R1') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_results_dir = save_results_dir
        self.phase = phase # If validation, we use client to evaluate
        self.format_mode = format_mode 

    def __call__(self, data: DataProto, name=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.save_results_dir is not None:
            sequences_data = []

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreA_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extracted_answer_list = []

        already_print_data_sources = {}

        import multiprocessing
        result_list = []
        with multiprocessing.Pool(processes=32) as pool:

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts'] # include left padding

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:] # remove left padding

                response_ids = data_item.batch['responses'] # include right padding
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length] # remove right padding

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                data_source = data_item.non_tensor_batch['data_source']

                extra_info = data_item.non_tensor_batch.get('extra_info', None)


                ### START
                compute_score_args = (data_source,
                    self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                    ground_truth,
                    extra_info,
                    self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                    self.phase,
                    self.format_mode,
                )

                #  acc, format_score, extracted_answer, from_judge, judge_response, judge_prompt
                # print('@Run----', flush=True)
                result = pool.apply_async(self.compute_score, args=compute_score_args)
                result_list.append((result, i, data_source, sequences_str, ground_truth))

                #### END
            pool.close()
            pool.join()

        for (result, i, data_source, sequences_str, ground_truth) in result_list:
            acc, format_score, extracted_answer, from_judge, judge_response = result.get()
            score = 0.9 * acc + 0.1 * format_score
            reward_tensor[i, valid_response_length - 1] = score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            acc_tensor[i, valid_response_length - 1] = acc

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"{data_source=} {sequences_str=} {extracted_answer=} {ground_truth=} {score=} {acc=} {format_score=}", flush=True)
            if self.save_results_dir is not None:
                sequences_data.append({'data_source': data_source, 'sequences_str': sequences_str, 
                                    'extracted_answer': extracted_answer, 'ground_truth': str(ground_truth), 
                                    'score': score, 'format_score': format_score, "acc": acc, 
                                    "from_judge": from_judge, "judge_response": judge_response}) #, "judge_prompt": judge_prompt})
            extracted_answer_list.append(extracted_answer)


        if self.save_results_dir is not None:
            df = pd.DataFrame(sequences_data)
            os.makedirs(self.save_results_dir, exist_ok=True)
            now = datetime.now()
            formatted_date = now.strftime("%Y_%m%d_%H_%M")
            try:
                if name is None:
                    save_path = os.path.join(self.save_results_dir, f'{formatted_date}.csv')
                else:
                    save_path = os.path.join(self.save_results_dir, f'{name}.csv')
                df.to_csv(save_path, escapechar='\\')


                print(f"In naive reward manager: {len(df)=}")

                df_excel = pd.read_csv(save_path, encoding='utf-8')
                excel_path = save_path.replace('.csv', '.xlsx')
                print(f"{excel_path=}")
                df_excel.to_excel(excel_path, index=False, engine='xlsxwriter')
            except Exception as e:
                print(f"Error saving CSV file: {e}")

        return reward_tensor, acc_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list
