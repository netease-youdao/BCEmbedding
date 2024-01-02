'''
@Description: 
@Author: shenlei
@Date: 2023-12-29 17:09:31
@LastEditTime: 2023-12-31 01:02:48
@LastEditors: shenlei
'''
import json
import os.path as osp
import pandas as pd

from typing import List

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever

from datasets import load_dataset

# Prompt to generate questions
qa_generate_prompt_tmpl_en = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
Generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""

qa_generate_prompt_tmpl_zh = """\
以下是上下文信息。

---------------------
{context_str}
---------------------

深入理解上述给定的上下文信息，而不是你的先验知识，根据下面的要求生成问题。

要求：你是一位教授，你的任务是为即将到来的考试设置{num_questions_per_chunk}个问题。你应该严格基于上下文信息，来设置多种多样的问题。\
你设置的问题不要包含选项，也不要以“问题1”或“问题2”为开头。\
将问题限制在所提供的上下文信息中。\
"""

def load_dataset_from_huggingface(dataset_name='maidalun1020/CrosslingualRAGDataset'):
    datasets_raw = load_dataset(dataset_name, split='dev')
    datasets = {}
    for dataset_raw in datasets_raw:
        for k in dataset_raw:
            dataset_raw[k] = json.loads(dataset_raw[k])
        pdf_file = dataset_raw.pop('pdf_file')
        datasets[pdf_file] = dataset_raw
    return datasets

# function to clean the dataset
def filter_qa_dataset(qa_dataset):
    # Extract keys from queries and relevant_docs that need to be removed
    queries_relevant_docs_keys_to_remove = {
        k for k, v in qa_dataset.queries.items()
        if 'Here are 2' in v or 'Here are two' in v
    }

    # Filter queries and relevant_docs using dictionary comprehensions
    filtered_queries = {
        k: v for k, v in qa_dataset.queries.items()
        if k not in queries_relevant_docs_keys_to_remove
    }
    filtered_relevant_docs = {
        k: v for k, v in qa_dataset.relevant_docs.items()
        if k not in queries_relevant_docs_keys_to_remove
    }

    # Create a new instance of EmbeddingQAFinetuneDataset with the filtered data
    return EmbeddingQAFinetuneDataset(
        queries=filtered_queries,
        corpus=qa_dataset.corpus,
        relevant_docs=filtered_relevant_docs
    )

def display_results(embedding_name, reranker_name, eval_results):

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = 100*full_df["hit_rate"].mean()
    mrr = 100*full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Embedding": [embedding_name], "Reranker": [reranker_name], "hit_rate": [hit_rate], "mrr": [mrr], "nums": [len(eval_results)]}
    )

    return metric_df

def extract_data_from_pdf(pdf_paths, llm=None, chunk_size=512, split=[0,36], qa_generate_prompt_tmpl=qa_generate_prompt_tmpl_en):
    if isinstance(pdf_paths, str) and osp.exists(pdf_paths):
        pdf_paths = [pdf_paths]
    documents = SimpleDirectoryReader(input_files=pdf_paths).load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents[split[0]:split[1]])

    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
    )
    # filter out pairs with phrases `Here are 2 questions based on provided context`
    qa_dataset = filter_qa_dataset(qa_dataset)

    return nodes, qa_dataset

# Define Retriever
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        reranker: None,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._reranker = reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)

        if self._reranker is None:
            retrieved_nodes = retrieved_nodes[:5]
        else:
            retrieved_nodes = self._reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        return retrieved_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aretrieve(str_or_query_bundle)