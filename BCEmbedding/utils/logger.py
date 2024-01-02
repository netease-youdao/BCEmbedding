'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 17:29:01
@LastEditTime: 2023-12-28 18:09:24
@LastEditors: shenlei
'''
import logging

def logger_wrapper(name='BCEmbedding'):
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger
