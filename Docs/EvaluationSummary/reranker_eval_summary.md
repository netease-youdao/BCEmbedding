<!--
 * @Description: 
 * @Author: shenlei
 * @Date: 2023-12-27 12:50:02
 * @LastEditTime: 2024-01-13 00:03:47
 * @LastEditors: shenlei
-->
# Reranker Evaluation Results  
## Language: `en`  

### Task Type: Reranking  
| Model | AskUbuntuDupQuestions | MindSmallReranking | SciDocsRR | StackOverflowDupQuestions | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|  
| bge-reranker-base | 54.70 | 28.48 | 67.09 | 37.55 | 46.96 |  
| bge-reranker-large | 58.73 | 28.84 | 71.30 | 39.04 | 49.48 |  
| bce-reranker-base_v1 | 56.54 | 30.73 | 75.79 | 42.88 | 51.48 |  

### *Summary on `en`*  
| Model | Reranking | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 46.96 | 46.96 |  
| bge-reranker-large | 49.48 | 49.48 |  
| bce-reranker-base_v1 | 51.48 | 51.48 |  
## Language: `zh`  

### Task Type: Reranking  
| Model | T2Reranking | MMarcoReranking | CMedQAv1 | CMedQAv2 | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|  
| bge-reranker-base | 67.28 | 35.46 | 81.27 | 84.10 | 67.03 |  
| bge-reranker-large | 67.60 | 37.64 | 82.14 | 84.18 | 67.89 |  
| bce-reranker-base_v1 | 70.25 | 34.13 | 79.64 | 81.31 | 66.33 |  

### *Summary on `zh`*  
| Model | Reranking | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 67.03 | 67.03 |  
| bge-reranker-large | 67.89 | 67.89 |  
| bce-reranker-base_v1 | 66.33 | 66.33 |  
## Language: `en-zh`  

### Task Type: Reranking  
| Model | T2RerankingEn2Zh | MMarcoRerankingEn2Zh | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|:--------:|  
| bge-reranker-base | 60.45 | 64.41 | 62.43 |  
| bge-reranker-large | 61.64 | 67.17 | 64.41 |  
| bce-reranker-base_v1 | 63.63 | 67.92 | 65.78 |  

### *Summary on `en-zh`*  
| Model | Reranking | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 62.43 | 62.43 |  
| bge-reranker-large | 64.41 | 64.41 |  
| bce-reranker-base_v1 | 65.78 | 65.78 |  
## Language: `zh-en`  

### Task Type: Reranking  
| Model | T2RerankingZh2En | MMarcoRerankingZh2En | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|:--------:|  
| bge-reranker-base | 63.94 | 63.79 | 63.87 |  
| bge-reranker-large | 64.13 | 67.89 | 66.01 |  
| bce-reranker-base_v1 | 65.38 | 67.23 | 66.31 |  

### *Summary on `zh-en`*  
| Model | Reranking | ***AVG*** |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 63.87 | 63.87 |  
| bge-reranker-large | 66.01 | 66.01 |  
| bce-reranker-base_v1 | 66.31 | 66.31 |  
## Summary on all langs: `['en', 'zh', 'en-zh', 'zh-en']`  
| Model | Reranking (12) | ***AVG*** (12) |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 59.04 | 59.04 |  
| bge-reranker-large | 60.86 | 60.86 |  
| bce-reranker-base_v1 | 61.29 | 61.29 |  
