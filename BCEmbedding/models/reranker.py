'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2023-12-28 18:07:46
@LastEditors: shenlei
'''
import logging
import torch

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('RerankerModel')


class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/yd-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = device
        assert self.device in ['cpu', 'cuda'], "Please input valid device: 'cpu' or 'cuda'!"
        self.num_gpus = 0 if self.device == "cpu" else num_gpus

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")
    
    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], 
            batch_size: int = 256,
            max_length: int = 512,
            enable_tqdm: bool=True,
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                scores_collection.extend(scores.cpu().numpy().tolist())
        
        if len(scores_collection) == 1:
            return scores_collection[0]
        return scores_collection

    def rerank(
            self,
            query: str,
            passages: List[str],
            **kwargs
        ):
        sentence_pairs = [[query, passage] for passage in passages]
        scores = self.compute_score(sentence_pairs, **kwargs)
        scores_argsort = np.argsort(scores)[::-1]

        sorted_scores = []
        sorted_passages = []
        for pid in scores_argsort:
            sorted_scores.append(scores[pid])
            sorted_passages.append(passages[pid])
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores
        }


