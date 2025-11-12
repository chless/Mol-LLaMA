import os
from typing import Any, Dict
import json
import re
from collections import defaultdict

import torch
from torch import optim
import pytorch_lightning as pl
from torch_geometric.data import Batch

from trainer.optims import LinearWarmupCosineLRScheduler
import torch.distributed as dist

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class MoleculeQATrainer(pl.LightningModule):
    def __init__(self, mol_llama, train_config):
        super().__init__()
        self.train_config = train_config

        self.mol_llama = mol_llama

        self.test_step_outputs = []

    def load_from_ckpt(self, ckpt_path):
        self.mol_llama.load_from_ckpt(ckpt_path)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.mol_llama(graph_batch, text_batch)
        loss = {'loss': output['loss']}

        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.mol_llama.generate(
            graph_batch, 
            text_batch,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
        pattern = r"[Aa]nswer:"

        # Generate further if the output does not contain "Answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nAnswer: ")
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = self.tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(self.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.device)

            new_responses = self.mol_llama.generate(
                new_graph_batch, 
                new_text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
                max_new_tokens=self.data_config.max_new_tokens,
            )
            new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]



        for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
            self.test_step_outputs.append({
                'response': response,
                'answer': answer,
                'task': task
            })

        return responses

    def on_validation_epoch_end(self):
        outputs = self.test_step_outputs

        if len(self.train_config.devices) > 1:
            all_outputs = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [item for sublist in all_outputs for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        texts = other_infos['answer']
        tasks = other_infos['task']
        input_texts = other_infos['input_text']

        outputs = self.mol_llama.generate(
            graph_batch, 
            text_batch,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
        )
        prediction_ids = outputs.sequences

        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        binary_classificaiton_probs = convert_logit2binary_prob(outputs.logits, self.tokenizer, tasks)

        save_dict = {
            'tasks': tasks,
            'input_texts': input_texts,
            'targets': texts,
            'predictions': predictions,
            'binary_classificaiton_probs': binary_classificaiton_probs,
        }
        # save tokenizer
        tokenizer_path = os.path.join(self.logger.log_dir)
        self.tokenizer.save_pretrained(tokenizer_path)

        self.save_predictions(**save_dict)

    def save_predictions(self, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        keys = list(kwargs.keys())
        len_dump = len(kwargs[keys[0]])
        for k in keys:
            assert len(kwargs[k]) == len_dump

        filepath = os.path.join(self.logger.log_dir, f'dumps_rank_{rank}.json')
        # load the previous dumps from filepath
        if os.path.exists(filepath):
            # read jsonl file
            with open(filepath, 'r', encoding='utf8') as f:
                cumulative_dumps = json.load(f)
        else:
            cumulative_dumps = []

        for i in range(len_dump):
            line = {k: kwargs[k][i] for k in keys}
            cumulative_dumps.append(line)

        with open(filepath, 'w', encoding='utf8') as f:
            # save json file
            json.dump(cumulative_dumps, f, ensure_ascii=True, indent=4)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        if len(self.train_config.devices) > 1:
            all_outputs = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [item for sublist in all_outputs for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, "test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, "test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
                

    def compute_metrics(self, outputs):
        results = defaultdict(list)
        corrects = {}

        for output in outputs:
            task = output['task']
            response = output['response']
            answer = output['answer'].replace("Answer: ", "")
            prediction = response.split("Answer: ")[-1].strip()
            if 'A' in prediction:
                prediction = 'A'
            elif 'B' in prediction:
                prediction = 'B'
            elif 'C' in prediction:
                prediction = 'C'
            elif 'D' in prediction:
                prediction = 'D'
            else:
                prediction = 'None'

            correct = 1 if prediction == answer else 0
            results[task].append({
                'response': response,
                'answer': answer,
                'prediction': prediction,
                'correct': correct
            })

            if task not in corrects:
                corrects[task] = {'correct': 0, 'total': 0}
            corrects[task]['correct'] += correct
            corrects[task]['total'] += 1

        tasks = list(corrects.keys())
        for task in tasks:
            correct = corrects[task]['correct']
            total = corrects[task]['total']
            accuracy = correct / total * 100
            corrects[task]['accuracy'] = accuracy

        # Calculate overall accuracy
        overall_correct = sum([corrects[task]['correct'] for task in tasks])
        overall_total = sum([corrects[task]['total'] for task in tasks])
        overall_accuracy = overall_correct / overall_total * 100
        corrects['overall'] = {'correct': overall_correct, 'total': overall_total, 'accuracy': overall_accuracy}

        return corrects, results

import torch

def convert_logit2binary_prob(logits, tokenizer, tasks):
    """Convert model logits into binary classification probabilities (True / False)."""

    classification_classes = {
        'bace',
        'smol-property_prediction-bbbp',
        'smol-property_prediction-clintox',
        'smol-property_prediction-hiv',
        'smol-property_prediction-sider',
    }

    # Create mask for which tasks are binary classification tasks
    classification_masks = [any(cls in task for cls in classification_classes) for task in tasks]
    classification_masks = torch.tensor(classification_masks, dtype=torch.bool, device=logits.device).unsqueeze(1)

    # Prepare token IDs (move them to same device as logits)
    positive_tokens = ["True", "true", "TRUE", "yes", "Yes", "YES"]
    negative_tokens = ["False", "false", "FALSE", "no", "No", "NO"]

    positive_token_ids = [tokenizer.encode(tok)[1] for tok in positive_tokens]
    negative_token_ids = [tokenizer.encode(tok)[1] for tok in negative_tokens]

    # Convert to tensors on same device
    positive_token_ids = torch.tensor(positive_token_ids, dtype=torch.long, device=logits.device)
    negative_token_ids = torch.tensor(negative_token_ids, dtype=torch.long, device=logits.device)

    # Compute probabilities
    probs = logits.softmax(dim=-1)
    batch_size, seq_len, _ = probs.size()

    false_logits = torch.zeros(batch_size, 1, device=logits.device)
    true_logits = torch.zeros(batch_size, 1, device=logits.device)
    target_logits_index = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    for i in range(batch_size):
        logits_i = logits[i]
        prediction_ids_i = logits_i.argmax(dim=-1)  # shape: [seq_len]

        # Find indices of any tokens matching positive/negative token IDs
        mask_match = torch.isin(prediction_ids_i, torch.cat((positive_token_ids, negative_token_ids)))

        if mask_match.any():
            first_match_idx = torch.nonzero(mask_match, as_tuple=False)[0, 0]
            target_logits_index[i] = first_match_idx
        else:
            target_logits_index[i] = 0

        # Use .sum() properly across the vocabulary dim
        false_logits[i] = probs[i, target_logits_index[i], negative_token_ids].sum()
        true_logits[i]  = probs[i, target_logits_index[i], positive_token_ids].sum()

    # Stack probabilities and normalize
    total_probs = torch.cat([false_logits, true_logits], dim=-1)
    total_probs = total_probs.softmax(dim=-1)

    # Fill non-classification tasks with -1
    total_probs = torch.where(classification_masks, total_probs, torch.full_like(total_probs, -1.0))

    # Convert to list for output
    return total_probs.tolist()