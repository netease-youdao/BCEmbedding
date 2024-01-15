'''
@Description: 
@Author: shenlei
@Date: 2024-01-15 18:56:59
@LastEditTime: 2024-01-15 23:18:38
@LastEditors: shenlei
'''
from __future__ import annotations

import torch

from typing import Dict, Optional, Sequence, Any

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from pydantic.v1 import PrivateAttr


def infer_torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class BCERerank(BaseDocumentCompressor):
    """Document compressor that uses `BCEmbedding RerankerModel API`."""

    client: str = 'BCEmbedding'
    top_n: int = 3
    """Number of documents to return."""
    model: str = "bce-reranker-base_v1"
    """Model to use for reranking."""
    _model: Any = PrivateAttr()

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        top_n: int = 3,
        model: str = "maidalun1020/bce-reranker-base_v1",
        device: Optional[str] = None,
        **kwargs
    ):
        try:
            from BCEmbedding.models import RerankerModel
        except ImportError:
            raise ImportError(
                "Cannot import `BCEmbedding` package,",
                "please `pip install BCEmbedding>=0.1.2`",
            )
        self._model = RerankerModel(model_name_or_path=model, device=device, **kwargs)
        super().__init__(top_n=top_n, model=model)

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["client"] = "BCEmbedding.models.RerankerModel"
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using `BCEmbedding RerankerModel API`.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)

        passages = []
        valid_doc_list = []
        invalid_doc_list = []
        for d in doc_list:
            passage = d.page_content
            if isinstance(passage, str) and len(passage) > 0:
                passages.append(passage.replace('\n', ' '))
                valid_doc_list.append(d)
            else:
                invalid_doc_list.append(d)

        rerank_result = self._model.rerank(query, passages)
        final_results = []
        for score, doc_id in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
            doc = valid_doc_list[doc_id]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        for doc in invalid_doc_list:
            doc.metadata["relevance_score"] = 0
            final_results.append(doc)

        final_results = final_results[:self.top_n]
        return final_results
