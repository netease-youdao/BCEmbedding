'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 18:52:01
@LastEditTime: 2023-12-28 18:08:52
@LastEditors: shenlei
'''
from typing import cast, List, Dict, Union

import numpy as np
from torch import nn
from BCEmbedding import EmbeddingModel


class YDDRESModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooler: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            batch_size: int = 160,
            max_length: int = 512,
            **kwargs
        ):
        self.model = EmbeddingModel(model_name_or_path=model_name_or_path, pooler=pooler, **kwargs)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
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
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.model.encode(
            input_texts,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            enable_tqdm=True
            )
    
    def encode(
            self,
            sentences: Union[str, List[str]],
            **kwargs
        ):
        return self.model.encode(
            sentences,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            enable_tqdm=True
        )


