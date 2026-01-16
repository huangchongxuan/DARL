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
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            # print(__file__)
            # print(f"{input_ids.shape=}")
            # print(f"{attention_mask.shape=}")
            # print(f"{position_ids.shape=}")

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature) # logits_rmpad.shape=torch.Size([23513, 152064])

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)


            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs


    def _forward_micro_batch_2(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # logits = output.logits
                # logits.div_(temperature)
                # logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                # logits = logits_rmpad

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                # print(f"logits_rmpad.shape: {logits_rmpad.shape}") # torch.Size([7462, 152064])
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    print(f"In use_ulysses_sp: {log_probs.shape}")
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    # raise NotImplementedError #("We have not implemented for all log probs")
                    assert False
                    # all_logits = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # print(f"1 log_probs.shape: {log_probs.shape}") # torch.Size([7462])
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                all_logits = pad_input(hidden_states=logits_rmpad.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                # print(f"full_log_probs.shape: {full_log_probs.shape}") # torch.Size([10, 1536, 1])
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                # print(f"log_probs.shape: {log_probs.shape}") # torch.Size([10, 1024])
                all_logits = all_logits.squeeze(-1)[:, -response_length - 1:-1]
                # print(f"all_logits.shape: {all_logits.shape}") # all_logits.shape: torch.Size([10, 1024, 152064])

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                # print(__file__)
                # print(f"logits.shape: {logits.shape}")
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                # print(f"log_probs.shape: {log_probs.shpae}")
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                all_logits = logits

            return entropy, log_probs, all_logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # self.actor_optimizer.step()
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            entropy_lst.append(entropy)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        # print(f"len(micro_batches) in compute_log_prob: {len(micro_batches)}") # 1

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            # print(f"log_probs.shape: {log_probs.shape}") # 10, 1024
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropy = entropy[revert_indices]

        return log_probs, entropy

    def compute_log_prob_pr(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses_pr', f'input_ids_pr', f'attention_mask_pr', f'position_ids_pr']
        batch = data.select(batch_keys=select_keys).batch
        for select_k in select_keys:
            batch[select_k.replace('_pr', '')] = batch.pop(select_k)

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            entropy_lst.append(entropy)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        # print(f"len(micro_batches) in compute_log_prob: {len(micro_batches)}") # 1

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            # print(f"log_probs.shape: {log_probs.shape}") # 10, 1024
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropy = entropy[revert_indices]

        return log_probs, entropy


    def compute_all_logits(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        # print(__file__)
        # print(f"data: {data}")
        # print(f"data.meta_info: {data.meta_info}")
        # print(f"data.keys: {data.keys()}")
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        all_logits_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, _, all_logits = self._forward_micro_batch_2(micro_batch, temperature=temperature)
            all_logits_lst.append(all_logits)
            # print(__file__)
            # print(f"micro_batch.shape: {micro_batch.shape}")  # torch.Size([10])
            # print('*'*50)
            # print(f"all_logits.shape: {all_logits.shape}")  # torch.Size([10, 1024, 152064])
            # print('*'*50)
        # print(f"micro_batches[0].shape: {micro_batches[0].shape}")  # micro_batches[0].shape: torch.Size([10])
        # print(f"len(micro_batches) in compute_all_logits: {len(micro_batches)}") # 1
        all_logits = torch.concat(all_logits_lst, dim=0) # [1, 1024, 152064]

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == all_logits.size(0), f"{len(indices)} vs. {all_logits.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            all_logits = all_logits[revert_indices]

        # for i in range(len(data)):
        #     data_item = data[i]
        #     ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        # print(__file__)
        # print(f"We save all_logits to CPU")
        all_logits = all_logits.to('cpu').float().numpy()

        return all_logits

    def update_policy(self, data: DataProto, mode='normal'): #, tokenizer_debug=None):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        if mode == 'normal':
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
            if self.config.use_kl_loss:
                select_keys.append('ref_log_prob')
            if self.config.get('use_sft_loss', False) and self.config.get('sft_type', 'bilevel') == 'multi_task':
                select_keys.extend(['ground_truth_mask_pr', 'old_log_probs_pr'])
            if 'response_mask_prob' in data.batch.keys(): # '1 + 2 = ' --> Model --> '<think> aaa </think> <answer> Model answer (GT) </answer>'
                # GT: 3
                # '1 + 2 = ' --> Model --> '<think> aaa </think> <answer> 123 </answer>' --> '<think> aaa </think> <answer> 3 </answer>' # 0.2
                # 0.2 is given to "<think> aaa </think>"
                # P(GT|Q + T)
                select_keys.append('response_mask_pr')
        elif mode == 'sft_only':
            assert self.config.get('use_sft_loss', False) and self.config.get('sft_type') == 'bilevel'
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                           'old_log_probs', 'ground_truth_mask_pr', 'old_log_probs_pr']
        else:
            raise NotImplementedError
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347

        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                if mode == 'normal':
                    responses = data['responses']
                    response_length = responses.size(1)
                    # We truncate the response mask here.
                    if 'response_mask_pr' in data: # if optimize think only
                        response_mask = data['response_mask_pr']
                    else:
                        attention_mask = data['attention_mask']
                        response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    if hasattr(self.config, "clip_ratio_low") and hasattr(self.config, "clip_ratio_high"):
                        clip_ratio = (self.config.clip_ratio_low, self.config.clip_ratio_high)
                    else:
                        clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.get('loss_agg_mode', 'token-mean')
                    max_tokens = self.config.get('max_tokens', None)

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                eos_mask=response_mask,
                                                                                cliprange=clip_ratio, 
                                                                                loss_agg_mode=loss_agg_mode,
                                                                                max_tokens=max_tokens)
                    # compute entropy loss from entropy
                    # entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    entropy_loss = core_algos.agg_loss(entropy, response_mask, loss_agg_mode=loss_agg_mode, max_tokens=max_tokens)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        # kl_loss = masked_mean(kld, response_mask)
                        kl_loss = core_algos.agg_loss(kld, response_mask, loss_agg_mode=loss_agg_mode, max_tokens=max_tokens)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    # Add SFT loss if enabled
                    if self.config.get('use_sft_loss', False) and self.config.get('sft_type', 'bilevel') == 'multi_task':
                        ground_truth_mask_pr = data['ground_truth_mask_pr']
                        old_log_prob_pr = data['old_log_probs_pr']
                        # Get log probs for ground truth tokens
                        eps = 1e-6
                        gt_old_log_prob_pr = ground_truth_mask_pr * old_log_prob_pr
                        token_mean_gt_old_log_prob_pr = gt_old_log_prob_pr.sum(dim=1) / (ground_truth_mask_pr.sum(dim=1) + eps)
                        sft_log_probs = token_mean_gt_old_log_prob_pr.sum() / (torch.count_nonzero(token_mean_gt_old_log_prob_pr) + eps)
                        # Compute negative log likelihood loss (cross entropy)
                        sft_loss = -sft_log_probs
                        policy_loss = policy_loss + sft_loss * self.config.sft_loss_coef
                        metrics['actor/sft_loss'] = sft_loss.detach().item()
                        metrics['actor/sft_coef'] = self.config.sft_loss_coef

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                elif mode == 'sft_only':
                    assert self.config.get('use_sft_loss', False) and self.config.get('sft_type') == 'bilevel'
                    ground_truth_mask_pr = data['ground_truth_mask_pr']
                    old_log_prob_pr = data['old_log_probs_pr']
                    # Get log probs for ground truth tokens
                    eps = 1e-6
                    gt_old_log_prob_pr = ground_truth_mask_pr * old_log_prob_pr
                    token_mean_gt_old_log_prob_pr = gt_old_log_prob_pr.sum(dim=1) / (ground_truth_mask_pr.sum(dim=1) + eps)
                    sft_log_probs = token_mean_gt_old_log_prob_pr.sum() / (torch.count_nonzero(token_mean_gt_old_log_prob_pr) + eps)
                    # Compute negative log likelihood loss (cross entropy)
                    sft_loss = -sft_log_probs
                    policy_loss = sft_loss * self.config.sft_loss_coef
                    metrics['actor/sft_loss'] = sft_loss.detach().item()
                    metrics['actor/sft_coef'] = self.config.sft_loss_coef
 
                else:
                    raise NotImplementedError

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

            grad_norm = self._optimizer_step()

            if mode == 'normal':
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
            elif mode == 'sft_only':
                data = {'acotr/sft_grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)

        self.actor_optimizer.zero_grad()
        return metrics
