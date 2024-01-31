<!--
 * @Description: 
 * @Author: shenlei
 * @Date: 2024-01-29 16:14:48
 * @LastEditTime: 2024-01-31 13:01:30
 * @LastEditors: shenlei
-->
# RAG Evaluations in LlamaIndex  

## Multiple Domains Scenarios in ["en"] 
| Embedding Models | WithoutReranker <br> [*hit_rate/mrr*] | CohereRerank <br> [*hit_rate/mrr*] | bge-reranker-large <br> [*hit_rate/mrr*] | ***bce-reranker-base_v1*** <br> [*hit_rate/mrr*] | 
|:-------------------------------|:--------:|:--------:|:--------:|:--------:| 
| OpenAI-ada-2 | 85.05/62.29 | 91.72/72.77 | 91.83/72.17 | **92.90/77.17** |  
| bge-large-en-v1.5 | 84.62/61.22 | 91.51/72.71 | 91.94/72.35 | **92.47/76.61** |  
| bge-m3-large | 86.67/64.22 | 92.15/73.19 | 92.69/72.04 | **93.33/77.24** |  
| llm-embedder | 77.53/56.10 | 86.34/69.36 | 86.56/68.78 | **87.42/73.44** |  
| CohereV3-en | 80.65/58.33 | 87.96/70.09 | 88.71/69.61 | **89.03/74.06** |  
| CohereV3-multilingual | 83.33/60.70 | 90.54/72.41 | 90.43/72.11 | **90.97/76.26** |  
| JinaAI-v2-Base-en | 81.94/58.03 | 90.32/71.65 | 90.75/71.11 | **91.29/75.53** |  
| gte-large-en | 83.44/59.18 | 90.97/72.24 | 91.61/72.38 | **92.26/76.52** |  
| e5-large-v2-en | 85.05/61.90 | 91.18/71.45 | 91.18/70.56 | **92.37/75.81** |  
| e5-large-multilingual | 85.91/61.87 | **93.01/73.38** | 92.80/72.45 | **93.44/77.31** |  
| ***bce-embedding-base_v1*** | **87.42/63.93** | 92.69/73.34 | **93.33/73.06** | ***93.87/77.88*** |  
