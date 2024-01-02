'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 17:21:15
@LastEditTime: 2024-01-02 14:12:59
@LastEditors: shenlei
'''
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import random

from tqdm import tqdm

from BCEmbedding import RerankerModel
from BCEmbedding.evaluation import c_mteb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="maidalun1020/bce-reranker-base_v1", type=str, help="model name or path you want to eval")
    parser.add_argument('--task_langs', default=['en', 'zh', 'en-zh', 'zh-en'], type=str, nargs='+', help="eval languages")  # default: monolingual, bilingual and crosslingual
    parser.add_argument('--use_fp16', default=False, action='store_true', help="eval in fp16 mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = RerankerModel(args.model_name_or_path, use_fp16=args.use_fp16)
    save_name = args.model_name_or_path.strip('/').split('/')[-1]
    
    tasks = [t for t in c_mteb.MTEB(task_types=["Reranking"], task_langs=args.task_langs).tasks]
    for task in tqdm(tasks, desc='RerankerModel Evaluating', unit='task'):
        evaluation = c_mteb.MTEB(
            tasks=[task], 
            task_langs=args.task_langs, 
        )
        evaluation.run(
            model, 
            output_folder=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/reranker_results", save_name),
            overwrite_results=False,
        )
