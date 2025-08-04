import sys
import numpy as np
import onnxruntime as ort
import torch
import pathlib
# eval
from typing import cast, List, Dict, Union
from collections import defaultdict
from typing import cast, List, Dict, Union
from mteb import MTEB
from mteb.tasks import SciFact, DuRetrieval

from transformers import AutoTokenizer

from datasets import DatasetDict, load_dataset

import random
random.seed(555)
import tqdm
from numpy import dot
from numpy.linalg import norm

tokenizer_path = "./tokenizer_bce"

cpu_onnx_model_path  = 'bce-embedding-base_v1.onnx'

onnx_model_path  = 'bce-embedding-base_v1_ctx.onnx'

device = "stx"
tensor_shape = "1x512"
tensor_dims = [int(d) for d in tensor_shape.split('x')]
print("onnx_model_path=", cpu_onnx_model_path)

print('INFO: Generating reference data using CPUExecutionProvider ...')
onnx_session_ref = ort.InferenceSession(cpu_onnx_model_path)


print('\nINFO: Reference data generated')

# Generate vai data
print('INFO: Creating onnx session using VitisAIExecutionProvider...')
cache_key = pathlib.Path(onnx_model_path).stem

onnx_session = ort.InferenceSession(
                onnx_model_path,                                                     
                providers=["VitisAIExecutionProvider"],                 
            )

def load_retrieval_data(dataset_path, dataset_revision, qrel_revision, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(dataset_path, revision=dataset_revision)
    qrels = load_dataset(dataset_path + "-qrels", revision=qrel_revision)[eval_split] # 9839, score only has 1

    g_sample = random.sample(range(len(qrels)), 100)
    c_sample = []
    q_sample = []
    for i, e in enumerate(qrels):
        if i in g_sample:
            c_sample.append(e['pid'])
            q_sample.append(e['qid'])
 

    corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"] if e['id'] in c_sample}
    queries = {e["id"]: e["text"] for e in dataset["queries"] if e['id'] in q_sample}

    # corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"]} # 100001
    # queries = {e["id"]: e["text"] for e in dataset["queries"]} # 2000
    
    relevant_docs = defaultdict(dict)
    for e in qrels:
        if e["qid"] not in queries or e["pid"] not in corpus:
            continue

        relevant_docs[e["qid"]][e["pid"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs

class Sampled_DuRetrieval(DuRetrieval):

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            "C-MTEB/DuRetrieval",
            "a1a333e290fe30b10f3f56498e3a0d911a693ced",
            "497b7bd1bbb25cb3757ff34d95a8be50a3de2279",
            ["dev"],
        )
        self.data_loaded = True

class MyModel:
    def __init__(
            self,
            session,
            tokenizer_path,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            batch_size: int = 256) -> None:
        self.model = session
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)


    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)


    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        all_embeddings = []

        for start_index in tqdm.tqdm(range(0, len(sentences), self.batch_size)):
            sentences_batch = sentences[start_index:start_index + self.batch_size]

            inputs = self.tokenizer(
                sentences_batch,
                padding='max_length',
                truncation=True,
                return_tensors='np',
                max_length=512,
            )
            
            last_hidden_state = self.model.run(None, {"input_ids":inputs['input_ids'].astype('int64') , "attention_mask":inputs['attention_mask'].astype('int64') })
            last_hidden_state = torch.tensor(last_hidden_state[0])
            embeddings = self.pooling(last_hidden_state, torch.tensor(inputs['attention_mask']))
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().float().numpy())

        vai_res = np.concatenate(all_embeddings, axis=0)

        ref_all_embeddings = []

        for start_index in tqdm.tqdm(range(0, len(sentences), self.batch_size)):
            sentences_batch = sentences[start_index:start_index + self.batch_size]

            inputs = self.tokenizer(
                sentences_batch,
                padding='max_length',
                truncation=True,
                return_tensors='np',
                max_length=512,
            )
            
            last_hidden_state = onnx_session.run(None, {"input_ids":inputs['input_ids'].astype('int64'), "attention_mask":inputs['attention_mask'].astype('int64')})
            last_hidden_state = torch.tensor(last_hidden_state[0])
            embeddings = self.pooling(last_hidden_state, torch.tensor(inputs['attention_mask']))
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            ref_all_embeddings.append(embeddings.cpu().float().numpy())
        cpu_res = np.concatenate(ref_all_embeddings, axis=0)
        cos_res = []
        l2_norm_res = []
        for a, b in zip(vai_res, cpu_res):
            cos_sim = dot(a,b)/(norm(a)*norm(b))
            cos_res.append(cos_sim)
            error_x_l2_norm = np.linalg.norm(a - b)
            error_x_l2_norm = error_x_l2_norm / np.linalg.norm(b) #np.size(pooled_BERTonnxCPU32)
            l2_norm_res.append(error_x_l2_norm)
        print(np.mean(cos_res))
        print(np.mean(l2_norm_res))
        return vai_res

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

def test_onnx(session, tokenizer_path, name):

    pipeline = MyModel(session=session, tokenizer_path = tokenizer_path,
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", 
                    pooling_method='cls', batch_size=1)
    evaluation = MTEB(tasks=[Sampled_DuRetrieval()])
    results = evaluation.run(pipeline, output_folder=f"zh_results/{name}") # , eval_splits=['test'])
    print(results)
    for key in results:
        print(key, 'ndcg_at_10:', results[key]['dev']['ndcg_at_10'])
    return



test_onnx(onnx_session_ref, tokenizer_path, 'ref')