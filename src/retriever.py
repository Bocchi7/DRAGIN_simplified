from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search

class BM25:
    def __init__(
        self,
        index_name: str = "wiki",
    ):
        self.max_ret_topk = 1000
        self.index_name = index_name
        self.es = BM25Search(
            index_name=index_name, 
            hostname='localhost', 
            initialize=False,
            number_of_shards=1).es
        
    def lexical_search(
        self,
        text: str,
        top_hits: int,
        skip: int = 0
    ) -> List[Tuple[str, float, str]]:
        # 基本照抄self.es.lexical_search，但是输出改一下
        req_body = {"query" : {"multi_match": {
                "query": text, 
                "type": "best_fields",
                "fields": [self.es.text_key, self.es.title_key],
                "tie_breaker": 0.5
                }}}
        
        res = self.es.es.search(
            search_type="dfs_query_then_fetch",
            index = self.index_name, 
            body = req_body, 
            size = skip + top_hits
        )
        
        hits = []
        
        for hit in res["hits"]["hits"][skip:]:
            hits.append(hit['_source']['txt']) # 只要文本
        
        return hits

    def __call__(
        self, 
        qry: str,
        topk: int = 1,
    ):
        result = self.lexical_search(
            text=qry, 
            top_hits=topk
        )
        return result