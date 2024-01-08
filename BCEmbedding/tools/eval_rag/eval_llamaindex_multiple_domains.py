'''
@Description: 
@Author: shenlei
@Date: 2023-12-26 16:24:57
@LastEditTime: 2024-01-09 01:04:24
@LastEditors: shenlei
'''
import os, json, sys
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

from utils import (qa_generate_prompt_tmpl_en, 
                    qa_generate_prompt_tmpl_zh, 
                    extract_data_from_pdf, display_results, 
                    CustomRetriever,
                    load_dataset_from_huggingface
                )

from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('tools.eval_rag.eval_llamaindex_multiple_domains')

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
    "bge-large-en-v1.5": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/bge-large-en-v1.5', 'device': 'cuda:0'}},
    "bge-large-zh-v1.5": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/bge-large-zh-v1.5', 'device': 'cuda:0'}},
    "llm-embedder": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'BAAI/llm-embedder', 'device': 'cuda:0'}},
    "CohereV3-en": {'model': CohereEmbedding, 'args': {'cohere_api_key': os.environ.get('COHERE_APPKEY'), 'model_name': 'embed-english-v3.0', 'input_type': 'search_document'}},
    "CohereV3-multilingual": {'model': CohereEmbedding, 'args': {'cohere_api_key': os.environ.get('COHERE_APPKEY'), 'model_name': 'embed-multilingual-v3.0', 'input_type': 'search_document'}},
    "JinaAI-v2-Base-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'jinaai/jina-embeddings-v2-base-en', 'pooling': 'mean', 'trust_remote_code': True, 'device': 'cuda:0'}},
    "gte-large-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'thenlper/gte-large', 'pooling': 'mean', 'max_length':512, 'device': 'cuda:0'}},
    "gte-large-zh": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'thenlper/gte-large-zh', 'max_length':512, 'device': 'cuda:0'}},
    "e5-large-v2-en": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'intfloat/e5-large-v2', 'pooling': 'mean', 'query_instruction': 'query:', 'text_instruction': 'passage:', 'device': 'cuda:0'}},
    "e5-large-multilingual": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'intfloat/multilingual-e5-large', 'pooling': 'mean', 'max_length': 512, 'query_instruction': 'query:', 'text_instruction': 'passage:', 'device': 'cuda:0'}},
    "bce-embedding-base_v1": {'model': HuggingFaceEmbedding, 'args': {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length':512, 'device': 'cuda:0'}},

}

RERANKERS = {
    "WithoutReranker": None,
    "CohereRerank": {'model': CohereRerank, 'args': {'api_key': os.environ.get('COHERE_APPKEY'), 'top_n': 5}},
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

pdfs_chunk_setup = {
    'Bio_en_0.pdf': {'chunk_size': 450, 'split': [1,21]},
    'Bio_zh_0.pdf': {'chunk_size': 512, 'split': [1,6]},

    'Comp_en_0.pdf': {'chunk_size': 450, 'split': [0,7]},
    'Comp_en_llama2.pdf': {'chunk_size': 450, 'split': [0,36]},
    'Comp_zh_0.pdf': {'chunk_size': 512, 'split': [0,3]},

    'Enco_en_0.pdf': {'chunk_size': 450, 'split': [0,34]},
    'Enco_zh_0.pdf': {'chunk_size': 512, 'split': [0,8]},

    'Math_en_0.pdf': {'chunk_size': 450, 'split': [0,7]},

    'Phys_en_0.pdf': {'chunk_size': 450, 'split': [1,19]},
    'Phys_zh_0.pdf': {'chunk_size': 512, 'split': [0,5]},

    'Q-fin_en_0.pdf': {'chunk_size': 450, 'split': [0,12]},
    'Q-fin_zh_0.pdf': {'chunk_size': 512, 'split': [2,13]},
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_pdfs_dir', default=osp.join(osp.dirname(__file__), 'eval_pdfs'), type=str, help="pdfs to eval")
    parser.add_argument('--llm', default='gpt-4-1106-preview', choices=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'], type=str, help="llm model name used in llama_index")
    parser.add_argument('--metrics', default=["mrr", "hit_rate"], type=str, nargs='+', help="eval metrics")
    parser.add_argument('--disable_dataset_from_huggingface', default=False, action='store_true', help="whether to use huggingface dataset for evaluation")
    parser.add_argument('--force', default=False, action='store_true', help="force to eval")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logger.info(args.__dict__)

    eval_pdfs_dir = args.eval_pdfs_dir
    eval_pdfs_cache_dir = osp.join(osp.dirname(__file__), 'cache/eval_pdfs_cache')
    eval_pdfs_results_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'results/rag_results')
    os.makedirs(eval_pdfs_cache_dir, exist_ok=True)
    os.makedirs(eval_pdfs_results_dir, exist_ok=True)

    llm = OpenAI(model=args.llm, api_key=os.environ.get('OPENAI_API_KEY'), api_base=os.environ.get('OPENAI_BASE_URL'))  # 'gpt-4-1106-preview', 'gpt-3.5-turbo-0613'

    if not args.disable_dataset_from_huggingface:
        datasets = load_dataset_from_huggingface()

    for pdf in os.listdir(eval_pdfs_dir):
        if pdf not in pdfs_chunk_setup:
            continue
        logger.info(40*'==' + f"\nEval {pdf} ...")
        pdf_path = osp.join(eval_pdfs_dir, pdf)
        pdf_cache = osp.join(eval_pdfs_cache_dir, pdf+'.json')
        pdf_eval_result = osp.join(eval_pdfs_results_dir, pdf+'.csv')

        # use preproc rag dataset on huggingface
        # NOTE: We preproc dataset has been translated (Youdao translation engine) to crosslingual queries for crosslingual setting, resulting in ['en-en', 'zh-zh', 'en-zh', 'zh-en'].
        if not args.disable_dataset_from_huggingface and pdf+'.json' in datasets:
            logger.info('load preproc rag dataset from huggingface')
            eval_data_cache = datasets[pdf+'.json']
            # load nodes and qa_dataset
            nodes = [TextNode.from_dict(node) for node in eval_data_cache['nodes']]
            qa_dataset = EmbeddingQAFinetuneDataset(
                            queries=eval_data_cache['queries'],
                            corpus=eval_data_cache['corpus'],
                            relevant_docs=eval_data_cache['relevant_docs']
                        )
        # use preproc local rag dataset
        elif osp.exists(pdf_cache):
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
        # use rag dataset produce online
        # NOTE: This procedure will produce bilingual datasets with ['en-en', 'zh-zh'] setting.
        else:
            logger.info('produce rag dataset from pdf')
            chunk_size = pdfs_chunk_setup[pdf]['chunk_size']
            split = pdfs_chunk_setup[pdf]['split']
            qa_generate_prompt_tmpl = qa_generate_prompt_tmpl_en if '_en_' in pdf else qa_generate_prompt_tmpl_zh
            nodes, qa_dataset = extract_data_from_pdf(pdf_path, llm=llm, chunk_size=chunk_size, split=split, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
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
