from transformers import AutoModelForCausalLM
from transformers import Trainer
import transformers.data.data_collator
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
import datasets

class LlamaTrainer(Trainer):
    def __init__(
            self,
            original_path: str = None,
            harmful_threshold : float = None,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: transformers.TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[..., Any]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ):
        super(LlamaTrainer,self).__init__(model, args, data_collator, train_dataset, eval_dataset,tokenizer, model_init,compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.original_model = AutoModelForCausalLM.from_pretrained(
            original_path,
            load_in_8bit=True,
            device_map='auto',
        )
        self.original_model.eval()
        self.harmful_threshold = harmful_threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        inputs_harm = {
            "input_ids": inputs['input_ids'],
            "labels": inputs['labels'],
            "attention_mask": inputs['attention_mask']
        }
        inputs_help = {
            "input_ids": inputs['input_ids2'],
            "labels": inputs['labels2'],
            "attention_mask": inputs['attention_mask2']
        }
        inputs_algn = {
            "input_ids": inputs['input_ids3'],
            "labels": inputs['labels3'],
            "attention_mask": inputs['attention_mask3']
        }

        outputs_harm = model(**inputs_harm)
        outputs_help = model(**inputs_help)
        outputs_algn = model(**inputs_algn)

        with torch.no_grad():
            outputs_help2 = self.original_model(**inputs_help).logits
            outputs_algn2 = self.original_model(**inputs_algn).logits

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs_harm[self.args.past_index]
        
        help_max = torch.max(outputs_help2).item()
        algn_max = torch.max(outputs_algn2).item()
        loss_harm = -1 * outputs_harm["loss"] if isinstance(outputs_harm, dict) else -1 * outputs_harm[0]
        loss_harm = torch.max(torch.tensor(0,dtype=torch.float32).cuda(), self.harmful_threshold + loss_harm)
        
        if math.isnan(help_max) and math.isnan(algn_max):
            loss = loss_harm
        elif math.isnan(help_max) and not math.isnan(algn_max):
            batch = outputs_algn.logits.shape[0]*outputs_algn.logits.shape[1]
            loss_algn = torch.nn.functional.kl_div(input=torch.nn.functional.log_softmax(outputs_algn.logits, dim=-1).reshape(batch,-1), target=torch.nn.functional.log_softmax(outputs_algn2, dim=-1).reshape(batch,-1).detach(), log_target=True, reduction='batchmean')
            loss_help = torch.tensor(0,dtype=torch.float32).cuda()
            loss = loss_harm + loss_algn
        elif not math.isnan(help_max) and math.isnan(algn_max):
            batch = outputs_help.logits.shape[0]*outputs_help.logits.shape[1]
            loss_help = torch.nn.functional.kl_div(input=torch.nn.functional.log_softmax(outputs_help.logits, dim=-1).reshape(batch,-1), target=torch.nn.functional.log_softmax(outputs_help2, dim=-1).reshape(batch,-1).detach(), log_target=True, reduction='batchmean')
            loss_algn = torch.tensor(0,dtype=torch.float32).cuda()
            loss = loss_harm + loss_help
        else:
            batch = outputs_algn.logits.shape[0]*outputs_algn.logits.shape[1]
            loss_algn = torch.nn.functional.kl_div(input=torch.nn.functional.log_softmax(outputs_algn.logits, dim=-1).reshape(batch,-1), target=torch.nn.functional.log_softmax(outputs_algn2, dim=-1).reshape(batch,-1).detach(), log_target=True, reduction='batchmean')
            batch = outputs_help.logits.shape[0]*outputs_help.logits.shape[1]
            loss_help = torch.nn.functional.kl_div(input=torch.nn.functional.log_softmax(outputs_help.logits, dim=-1).reshape(batch,-1), target=torch.nn.functional.log_softmax(outputs_help2, dim=-1).reshape(batch,-1).detach(), log_target=True, reduction='batchmean')
            loss = loss_harm + loss_help + loss_algn
        print('loss_harm={}, loss_help={}, loss_algn={}, loss={}'.format(loss_harm.item(), loss_help.item(), loss_algn.item(), loss.item()))
        
        return (loss, outputs_harm) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        data_collator.signature_columns.append('input_ids2')
        data_collator.signature_columns.append('labels2')
        data_collator.signature_columns.append('attention_mask2')
        data_collator.signature_columns.append('input_ids3')
        data_collator.signature_columns.append('labels3')
        data_collator.signature_columns.append('attention_mask3')

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))