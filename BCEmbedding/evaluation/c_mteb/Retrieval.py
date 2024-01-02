from collections import defaultdict
from datasets import load_dataset, DatasetDict

from mteb import AbsTaskRetrieval
from mteb.tasks import load_retrieval_data

class CrosslingualRetrievalBooksEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalBooksEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalBooksEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalBooksZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalBooksZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalBooksZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalFinanceEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalFinanceEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalFinanceEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalFinanceZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalFinanceZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalFinanceZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalLawEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalLawEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalLawEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalLawZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalLawZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalLawZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalOthersEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalOthersEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalOthersEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalOthersZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalOthersZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalOthersZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalPaperEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalPaperEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalPaperEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalPaperZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalPaperZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalPaperZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalWikiEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalWikiEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalWikiEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalWikiZh2En(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalWikiZh2En',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalWikiZh2En',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CrosslingualRetrievalQasEn2Zh(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CrosslingualRetrievalQasEn2Zh',
            'hf_hub_name': 'maidalun1020/CrosslingualRetrievalQasEn2Zh',
            'reference': '',
            'description': '',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'ndcg_at_3',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True