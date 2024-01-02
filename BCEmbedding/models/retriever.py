'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 18:48:11
@LastEditTime: 2023-12-28 18:07:52
@LastEditors: shenlei
'''
import faiss
import numpy as np
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('RetrieverModel')


class RetrieverModel:
    def __init__(self, device: str='cpu', num_cells_in_search: int=10):
        self.device = device
        self.num_cells_in_search = num_cells_in_search
        self.reset_index()
    
    def reset_index(self):
        self.index = {'seen': set()}
    
    def build_index(self, passages: Union[str, List[str]], embeddings: ndarray):
        if isinstance(passages, str):
            passages = [passages]
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)
            
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim > 2:
            logger.error("`embeddings.ndim` should be 2, which denotes num of embeddings and dim of embedding vector.")
            return False

        if len(passages) != embeddings.shape[0]:
            logger.error("num of passages not equal to num of embeddings.")
            return False
        
        new_passages = []
        new_embeddings = []
        for pid, passage in enumerate(passages):
            if passage not in self.index['seen']:
                self.index['seen'].add(passage)
                new_passages.append(passage)
                new_embeddings.append(embeddings[pid])
        new_embeddings = np.asarray(new_embeddings)

        if 'index' not in self.index:
            index = faiss.IndexFlatIP(new_embeddings.shape[1])
            if self.device == 'cuda' and hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                res.setTempMemory(16 * 1024 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(new_embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(new_embeddings))
            self.index["index"] = index
            self.index['passages'] = new_passages
        else:
            logger.warning("Now you add new embeddings to existing faiss index. If you deal with new dataset, please `{RetrievalModel}.reset_index()` first!")
            self.index["index"].add(new_embeddings.astype(np.float32))
            self.index["passages"].extend(new_passages)
        return True

    def search(self, embeddings: ndarray, threshold: float=0.2, top_k: int=50):
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)
            
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim > 2:
            logger.error("`embeddings.ndim` should be 2, which denotes num of embeddings and dim of embedding vector.")
            return None
        
        if 'index' not in self.index:
            logger.error("Please build index first.")
            return None
        
        logger.info(f"Search top {top_k} results of {embeddings.shape[0]} queries ...")
        distance, idx = self.index["index"].search(embeddings.astype(np.float32), top_k)
        search_results = []
        for i in range(embeddings.shape[0]):
            results = self._pack_single_result(distance[i], idx[i], threshold)
            search_results.append(results)
        return search_results
    
    def _pack_single_result(self, dist: List, idx: List, threshold: float):
        results = [self.index["passages"][i] for i, s in zip(idx, dist) if s >= threshold and 0 <= i < len(self.index["passages"])]
        return results
    
    def evaluate(self, preds: List[List[str]], labels: Union[str, List[str], List[List[str]]], cutoffs: List=[5,20,50]):
        """
        Evaluate MRR and Recall at cutoffs.
        """
        labels_format = []
        if isinstance(labels, str):
            labels_format.append([labels])
        elif isinstance(labels, list):
            for label in labels:
                if isinstance(label, str):
                    labels_format.append([label])
                elif isinstance(label, list):
                    labels_format.append(list(set(label)))
        
        if len(preds) != len(labels_format):
            logger.error("Num of predictions should equal to num of labels")
            return None

        metrics = {}
        # MRR
        mrrs = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels_format):
            for i, x in enumerate(pred):
                if x in label:
                    for k, cutoff in enumerate(cutoffs):
                        if i < cutoff:
                            mrrs[k] += 1 / (i+1)
                    break
        mrrs /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            metrics["MRR@{:03d}".format(cutoff)] = mrrs[i]

        # Recall
        recalls = np.zeros(len(cutoffs))
        exact_recalls = 0
        for pred, label in zip(preds, labels_format):
            for k, cutoff in enumerate(cutoffs):
                recall = np.intersect1d(label, pred[:cutoff])
                recalls[k] += len(recall) / len(label)
            
            recall = np.intersect1d(label, pred[:len(label)])
            exact_recalls += len(recall)/len(label)

        recalls /= len(preds)
        exact_recalls /= len(preds)

        for i, cutoff in enumerate(cutoffs):
            metrics["Recall@{:03d}".format(cutoff)] = recalls[i]
        
        metrics["Recall@exact"] = exact_recalls

        return metrics
