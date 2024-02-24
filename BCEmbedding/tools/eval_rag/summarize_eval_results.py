'''
@Description: 
@Author: shenlei
@Date: 2023-12-31 01:04:18
@LastEditTime: 2024-02-06 10:33:56
@LastEditors: shenlei
'''
import os, json, sys
import os.path as osp
import argparse
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('tools.eval_rag.summarize_eval_results')
from IPython import embed

def read_results(results_dir):
    csv_files = [osp.join(results_dir, i) for i in os.listdir(results_dir) if i.endswith('.csv')]
    logger.info('find {} csv files'.format(len(csv_files)))
    tot_num = 0
    for csv_file in csv_files:
        csv_df = pd.read_csv(csv_file)
        tot_num += csv_df.values[0,-1]
    logger.info('total number of queries: {}'.format(tot_num))

    merged_results = OrderedDict()
    for csv_file in tqdm(csv_files):
        csv_df = pd.read_csv(csv_file)
        for _, it in csv_df.iterrows():
            queries_num = it['nums']
            embedding = it['Embedding']
            if embedding not in merged_results:
                merged_results[embedding] = OrderedDict()
            
            reranker = it['Reranker']
            if reranker not in merged_results[embedding]:
                merged_results[embedding][reranker] = {'hit_rate': 0, 'mrr': 0}
            
            merged_results[embedding][reranker]['hit_rate'] += it['hit_rate'] * (queries_num/tot_num)
            merged_results[embedding][reranker]['mrr'] += it['mrr'] * (queries_num/tot_num)
    
    return merged_results


def output_markdown(merged_results, save_file, eval_more_domains=True):
    with open(save_file, 'w') as f:
        f.write(f"# RAG Evaluations in LlamaIndex  \n")

        if eval_more_domains:
            f.write(f'\n## Multiple Domains Scenarios  \n')
        else:
            f.write(f'\n## Reproduce [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)  \n')

        first_line = "| Embedding Models |"
        second_line = "|:-------------------------------|"
        embeddings = sorted(list(merged_results.keys()))
        rerankers = []
        for embedding in embeddings:
            for reranker in list(merged_results[embedding].keys()):
                if reranker not in rerankers:
                    rerankers.append(reranker)

        for reranker in rerankers:
            first_line += f" {reranker} <br> [*hit_rate/mrr*] |"
            second_line += ":--------:|"
        f.write(first_line + ' \n')
        f.write(second_line + ' \n')

        for embedding in embeddings:
            embedding_results = merged_results[embedding]
            write_line = f"| {embedding} |"
            for reranker in rerankers:
                if reranker in embedding_results:
                    write_line += f" {embedding_results[reranker]['hit_rate']:.2f}/{embedding_results[reranker]['mrr']:.2f} |"
                else:
                    write_line += " -- |"
            f.write(write_line + '  \n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/rag_results"), type=str, help="pdfs to eval")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    eval_more_domains = len([i for i in os.listdir(args.results_dir) if i.endswith('.csv')]) > 1
    save_name = 'rag_eval_multiple_domains_summary' if eval_more_domains else 'rag_eval_reproduced_summary'
    save_path = os.path.join(args.results_dir, save_name + ".md")
    logger.info('Summary {}. Save as {}'.format('on more domains' if eval_more_domains else 'reproduce from BLOG', save_path))

    merged_results = read_results(args.results_dir)
    output_markdown(
        merged_results, 
        save_path,
        eval_more_domains
        )