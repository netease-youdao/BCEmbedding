'''
@Description: 
@Author: shenlei
@Date: 2023-12-26 16:24:57
@LastEditTime: 2024-01-09 01:05:12
@LastEditors: shenlei
'''
import os, json
import os.path as osp
import argparse
import pandas as pd

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.schema import TextNode

# LLM
from llama_index.llms import OpenAI

# Embeddings
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding, CohereEmbedding
from langchain.embeddings import VoyageEmbeddings, GooglePalmEmbeddings

# Retrievers
from llama_index.retrievers import VectorIndexRetriever

# Rerankers
from llama_index.postprocessor import CohereRerank
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

# Evaluator
from llama_index.evaluation import RetrieverEvaluator

from utils import qa_generate_prompt_tmpl_en, extract_data_from_pdf, display_results, CustomRetriever

from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('tools.eval_rag.eval_llamaindex_reproduce')

import nest_asyncio
import asyncio
nest_asyncio.apply()

doc_string = '''
Hit Rate:
Hit rate calculates the fraction of queries where the correct answer is found within the top-k retrieved documents. In simpler terms, it's about how often our system gets it right within the top few guesses.

Mean Reciprocal Rank (MRR):
For each query, MRR evaluates the system's accuracy by looking at the rank of the highest-placed relevant document. Specifically, it's the average of the reciprocals of these ranks across all the queries. So, if the first relevant document is the top result, the reciprocal rank is 1; if it's second, the reciprocal rank is 1/2, and so on.
'''

# Define embeddings and rerankers setups to test
EMBEDDINGS = {
    "OpenAI-ada-2": {'model': OpenAIEmbedding, 'args': {'api_key': os.environ.get('OPENAI_API_KEY'), 'api_base': os.environ.get('OPENAI_BASE_URL')}},
    "bge-large-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/bge-large-en', 'device': 'cuda:0'}},
    "bge-base-en-v1.5": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/bge-base-en-v1.5', 'device': 'cuda:0'}},
    "bge-large-en-v1.5": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/bge-large-en-v1.5', 'device': 'cuda:0'}},
    "llm-embedder": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/llm-embedder', 'device': 'cuda:0'}},
    "CohereV2-en": {'model': CohereEmbedding, 'args': {'cohere_api_key': os.environ.get('COHERE_APPKEY'), 'model_name': 'embed-english-v2.0'}},
    "CohereV3-en": {'model': CohereEmbedding, 'args': {'cohere_api_key': os.environ.get('COHERE_APPKEY'), 'model_name': 'embed-english-v3.0', 'input_type': 'search_document'}},
    "JinaAI-v2-Small-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'jinaai/jina-embeddings-v2-small-en', 'pooling': 'mean', 'trust_remote_code': True, 'device': 'cuda:0'}},
    "JinaAI-v2-Base-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'jinaai/jina-embeddings-v2-base-en', 'pooling': 'mean', 'trust_remote_code': True, 'device': 'cuda:0'}},
    "gte-large-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'thenlper/gte-large', 'pooling': 'mean', 'max_length':512, 'device': 'cuda:0'}},
    "e5-large-v2-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'intfloat/e5-large-v2', 'pooling': 'mean', 'query_instruction': 'query:', 'text_instruction': 'passage:', 'device': 'cuda:0'}},
    "e5-base-multilingual": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'intfloat/multilingual-e5-base', 'pooling': 'mean', 'max_length': 512, 'query_instruction': 'query:', 'text_instruction': 'passage:', 'device': 'cuda:0'}},
    "e5-large-multilingual": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'intfloat/multilingual-e5-large', 'pooling': 'mean', 'max_length': 512, 'query_instruction': 'query:', 'text_instruction': 'passage:', 'device': 'cuda:0'}},
    "bce-embedding-base_v1": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length':512, 'device': 'cuda:0'}},
}

RERANKERS = {
    "WithoutReranker": None,
    "CohereRerank": {'model': CohereRerank, 'args': {'api_key': os.environ.get('COHERE_APPKEY'), 'top_n': 5}},
    "bge-reranker-base": {'model': SentenceTransformerRerank, 'args': {'model': "BAAI/bge-reranker-base", 'top_n': 5, 'device': 'cuda:1'}},
    "bge-reranker-large": {'model': SentenceTransformerRerank, 'args': {'model': "BAAI/bge-reranker-large", 'top_n': 5, 'device': 'cuda:1'}},
    "bce-reranker-base_v1": {'model': SentenceTransformerRerank, 'args': {'model': "maidalun1020/bce-reranker-base_v1", 'top_n': 5, 'device': 'cuda:1'}},
}

logger.info(f"""Evaluate with metrics in RAG framework:
{doc_string}

The evaluation processes {len(EMBEDDINGS)} embeddings and {len(RERANKERS)} rerankers:
embeddings: {list(EMBEDDINGS.keys())}
rerankers: {list(RERANKERS.keys())}
{40*'=='}
""")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_pdfs_dir', default=osp.join(osp.dirname(__file__), 'eval_pdfs'), type=str, help="pdfs to eval")
    parser.add_argument('--llm', default='gpt-3.5-turbo-0613', type=str, help="llm model name used in llama_index")
    parser.add_argument('--chunk_size', default=512, type=int, help="chunk size for splitting context of pdfs into chunks.")
    parser.add_argument('--split', default=[0, 36], type=int, nargs='+', help="choice page range of pdfs")
    parser.add_argument('--metrics', default=["mrr", "hit_rate"], type=str, nargs='+', help="eval metrics")
    parser.add_argument('--force', default=False, action='store_true', help="force to eval")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logger.info(args.__dict__)

    eval_pdfs_dir = args.eval_pdfs_dir
    eval_pdfs_cache_dir = osp.join(osp.dirname(__file__), 'cache/eval_pdfs_reproduce_cache')
    eval_pdfs_results_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'results/rag_reproduce_results')
    os.makedirs(eval_pdfs_cache_dir, exist_ok=True)
    os.makedirs(eval_pdfs_results_dir, exist_ok=True)

    llm = OpenAI(model=args.llm, api_key=os.environ.get('OPENAI_API_KEY'), api_base=os.environ.get('OPENAI_BASE_URL'))

    pdf = 'Comp_en_llama2.pdf'
    logger.info(40*'==' + f"\nEval {pdf} ...")
    pdf_path = osp.join(eval_pdfs_dir, pdf)
    pdf_cache = osp.join(eval_pdfs_cache_dir, pdf+'.json')
    pdf_eval_result = osp.join(eval_pdfs_results_dir, pdf+'.csv')

    if osp.exists(pdf_cache):
        logger.info('load preproc rag dataset from local')
        with open(pdf_cache, 'r') as f:
            eval_data_cache = json.load(f)
        # load nodes and qa_dataset
        nodes = [TextNode.from_dict(node) for node in eval_data_cache['nodes']]
        qa_dataset = EmbeddingQAFinetuneDataset(
                        queries=eval_data_cache['queries'],
                        corpus=eval_data_cache['corpus'],
                        relevant_docs=eval_data_cache['relevant_docs']
                    )
    else:
        logger.info('produce rag dataset from pdf')
        nodes, qa_dataset = extract_data_from_pdf(pdf_path, llm=llm, chunk_size=args.chunk_size, split=args.split, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl_en)
        # save nodes and qa_dataset
        eval_data_to_save = qa_dataset.__dict__
        eval_data_to_save['nodes'] = [node.dict() for node in nodes]
        with open(pdf_cache, 'w') as f:
            json.dump(eval_data_to_save, f, indent=4, ensure_ascii=False)

    # eval
    results_df = pd.read_csv(pdf_eval_result) if osp.exists(pdf_eval_result) else pd.DataFrame()
    for embed_name, embed_model_setup in EMBEDDINGS.items():
        embed_model = embed_model_setup['model'](**embed_model_setup['args'])
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        vector_index = VectorStoreIndex(nodes, service_context=service_context)

        if not embed_name.startswith('CohereV3'):
            vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10, service_context=service_context)
        else:
            embed_model = CohereEmbedding(cohere_api_key=os.environ.get('COHERE_APPKEY'), model_name=embed_model_setup['args']['model_name'], input_type='search_query')
            service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
            vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10, service_context=service_context)

        # Loop over rerankers
        for rerank_name, reranker_setup in RERANKERS.items():
            if not args.force:
                has_evaluated = False
                for _, it in results_df.iterrows():
                    if it['Embedding'] == embed_name and it['Reranker'] == rerank_name:
                        has_evaluated = True
                if has_evaluated:
                    logger.info(f"Skip! Embedding Model: {embed_name} and Reranker: {rerank_name} have been evaluated!")
                    continue

            logger.info('\n'+ 40*'-*' + f"\nRunning Evaluation for Embedding Model: {embed_name} and Reranker: {rerank_name}")
            
            reranker = None if reranker_setup is None else reranker_setup['model'](**reranker_setup['args'])
            custom_retriever = CustomRetriever(vector_retriever, reranker)
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                args.metrics, retriever=custom_retriever
            )

            eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(qa_dataset))

            current_df = display_results(embed_name, rerank_name, eval_results)
            results_df = pd.concat([results_df, current_df], ignore_index=True)

            logger.info(current_df)
            results_df.to_csv(pdf_eval_result, index=False)

    # Display final results
    logger.info(40*'-*' + '\nfinal summary:')
    logger.info(results_df)
    results_df.to_csv(pdf_eval_result, index=False)
