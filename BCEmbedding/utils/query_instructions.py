'''
@Description: 
@Author: shenlei
@Date: 2023-12-18 16:54:50
@LastEditTime: 2024-01-07 00:22:50
@LastEditors: shenlei
'''

query_instruction_for_retrieval_dict = {
    "BAAI/bge-large-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",

    "BAAI/bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-zh-noinstruct": None,
    "BAAI/bge-base-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh-v.15": "为这个句子生成表示以用于检索相关文章：",

    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-large-v2": "query: ",
}

passage_instruction_for_retrieval_dict = {
    "intfloat/multilingual-e5-base": "passage: ",
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
    "intfloat/e5-large-v2": "passage: ",
}