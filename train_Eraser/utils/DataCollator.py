from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional,  Union

import numpy as np

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        features_list = []
        for lab_name, inp_name, att_name in zip(['labels', 'labels2', 'labels3'],['input_ids', 'input_ids2', 'input_ids3'],['attention_mask', 'attention_mask2', 'attention_mask3']):
            labels = [feature[lab_name] for feature in features] if lab_name in features[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature[lab_name]))
                    if isinstance(feature[lab_name], list):
                        feature[lab_name] = (
                            feature[lab_name] + remainder if padding_side == "right" else remainder + feature[lab_name]
                        )
                    elif padding_side == "right":
                        feature[lab_name] = np.concatenate([feature[lab_name], remainder]).astype(np.int64)
                    else:
                        feature[lab_name] = np.concatenate([remainder, feature[lab_name]]).astype(np.int64)
            features_linshi = [{"labels":fea[lab_name], "input_ids":fea[inp_name], "attention_mask":fea[att_name]} for fea in features]
            features_list.append(self.tokenizer.pad(
                features_linshi,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            ))

        features = features_list[0]
        for i, fea in enumerate(features_list[1:]):
            features.data["input_ids{}".format(i+2)] = fea.data["input_ids"]
            features.data["labels{}".format(i+2)] = fea.data["labels"]
            features.data["attention_mask{}".format(i+2)] = fea.data["attention_mask"]
        return features


