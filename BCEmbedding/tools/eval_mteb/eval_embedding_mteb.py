'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 17:21:19
@LastEditTime: 2024-01-02 14:13:21
@LastEditors: shenlei
'''
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import random
from tqdm import tqdm

from BCEmbedding.evaluation import c_mteb, YDDRESModel

from BCEmbedding.utils import query_instruction_for_retrieval_dict
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.eval_embedding_mteb')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="maidalun1020/bce-embedding-base_v1", type=str, help="model name or path you want to eval")
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    parser.add_argument('--task_langs', default=['en', 'zh', 'zh-CN'], type=str, nargs='+', help="eval langs")  # default: monolingual, bilingual and crosslingual
    parser.add_argument('--pooler', default='cls', type=str, help="pooling method of embedding model, default `cls`.")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size to inference.")
    parser.add_argument('--use_fp16', default=False, action='store_true', help="eval in fp16 mode")
    parser.add_argument('--trust_remote_code', default=False, action='store_true', help="load huggingface model trust_remote_code")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = YDDRESModel(
        model_name_or_path=args.model_name_or_path,
        normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
        pooler=args.pooler,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        trust_remote_code=args.trust_remote_code,
        )
    save_name = args.model_name_or_path.strip('/').split('/')[-1]

    tasks = [t for t in c_mteb.MTEB(task_types=args.task_type, task_langs=args.task_langs).tasks]
    for task in tqdm(tasks, desc='EmbeddingModel Evaluating', unit='task'):
        if task.description["name"] in ['MSMARCOv2', 'BUCC']:
            logger.info('Skip task {}, which is too large to eval.'.format(task.description["name"]))
            continue

        if 'CQADupstack' in task.description["name"] or \
            'T2Reranking' in task.description["name"] or \
            'T2Retrieval' in task.description["name"] or \
            'MMarcoReranking' in task.description["name"] or \
            'MMarcoRetrieval' in task.description["name"] or \
            'CrosslingualRetrieval' in task.description["name"] or \
            task.description["name"] in [
                # en
                'Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', 
                
                # zh
                'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval',
                'CovidRetrieval', 'CmedqaRetrieval',
                'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
                'T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2'
            ]:
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                instruction = None
                logger.info(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
                logger.info(f"{args.model_name_or_path} in query_instruction_for_retrieval_dict, set instruction={instruction}")
        else:
            instruction = None

        model.query_instruction_for_retrieval = instruction
        evaluation = c_mteb.MTEB(
            tasks=[task], 
            task_langs=args.task_langs, 
        )

        evaluation.run(
            model, 
            output_folder=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/embedding_results", save_name),
            overwrite_results=False,
        )
