'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-05-13 17:04:23
@LastEditors: shenlei
'''
import logging
import torch

from tqdm import tqdm
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union

from transformers import AutoModel, AutoTokenizer
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('BCEmbedding.models.EmbeddingModel')


class EmbeddingModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/bce-embedding-base_v1',
            pooler: str='cls',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")

        assert pooler in ['cls', 'mean'], f"`pooler` should be in ['cls', 'mean']. 'cls' is recommended!"
        self.pooler = pooler
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16};\t embedding pooling type: {self.pooler};\t trust remote code: {kwargs.get('trust_remote_code', False)}")

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int=256,
            max_length: int=512,
            normalize_to_unit: bool=True,
            return_numpy: bool=True,
            enable_tqdm: bool=True,
            query_instruction: str="",
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        with torch.no_grad():
            embeddings_collection = []
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                if isinstance(query_instruction, str) and len(query_instruction) > 0:
                    sentence_batch = [query_instruction+sent for sent in sentences[sentence_id:sentence_id+batch_size]] 
                else:
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                        sentence_batch, 
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs_on_device, return_dict=True)

                if self.pooler == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "mean":
                    attention_mask = inputs_on_device['attention_mask']
                    last_hidden = outputs.last_hidden_state
                    embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embeddings_collection.append(embeddings.cpu())
            
            embeddings = torch.cat(embeddings_collection, dim=0)
        
        if return_numpy and not isinstance(embeddings, ndarray):
            embeddings = embeddings.numpy()
        
        return embeddings
