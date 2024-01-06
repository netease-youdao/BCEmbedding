'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 18:52:01
@LastEditTime: 2024-01-07 00:17:13
@LastEditors: shenlei
'''
from typing import cast, List, Dict, Union

import numpy as np
from torch import nn
from BCEmbedding import EmbeddingModel
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.c_mteb.yd_dres_model')


class YDDRESModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooler: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            passage_instruction_for_retrieval: str = None,
            batch_size: int = 160,
            max_length: int = 512,
            **kwargs
        ):
        self.model = EmbeddingModel(model_name_or_path=model_name_or_path, pooler=pooler, **kwargs)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length

        self.instruction_for_all = "e5-base" in model_name_or_path or "e5-large" in model_name_or_path

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        logger.info(f'##BCEmbedding##: using encode_queries with instruction: {self.query_instruction_for_retrieval}')
        return self.model.encode(
            queries,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            query_instruction=self.query_instruction_for_retrieval,
            enable_tqdm=True
            )

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        logger.info(f'##BCEmbedding##: using encode_corpus with instruction: {self.passage_instruction_for_retrieval}')
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.model.encode(
            input_texts,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            query_instruction=self.passage_instruction_for_retrieval,
            enable_tqdm=True
            )
    
    def encode(
            self,
            sentences: Union[str, List[str]],
            **kwargs
        ):
        if self.instruction_for_all:
            assert len(self.query_instruction_for_retrieval) > 0
            instruction = self.query_instruction_for_retrieval
        else:
            instruction = None
            
        logger.info(f'##BCEmbedding##: instruction for all: {self.instruction_for_all}; using encode with instruction: {instruction}')
        return self.model.encode(
            sentences,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            query_instruction=instruction,
            enable_tqdm=True
        )
