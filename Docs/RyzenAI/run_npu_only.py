# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

import sys
import json
import os
import pathlib
import argparse
import numpy as np
import onnxruntime as ort
import onnx


parser = argparse.ArgumentParser(description="onnxruntime compile and run using python API")
parser.add_argument("--input_model", "-i", help="Path to the model (.onnx for ONNX) for onnxruntime compile")
args = parser.parse_args()
onnx_model_path  = args.input_model
print("onnx_model_path=", onnx_model_path)

np.random.seed(123)
#Generate IFMs
batch_size = 1
sequence_length = 512
input_ids = np.random.randint(0, 2048, size=(batch_size, sequence_length), dtype=np.int64)
attention_mask = np.random.randint(0, 2048,size=(batch_size, sequence_length), dtype=np.int64)
ifm =  np.full((1,512), 1, dtype=np.int64)
input_data = {"input_ids": ifm, "attention_mask": ifm}

print('INFO: Creating onnx session using VitisAIExecutionProvider...')
cache_key = pathlib.Path(onnx_model_path).stem

import time
start = time.perf_counter()
onnx_session = ort.InferenceSession(
                   onnx_model_path,
                   providers=["VitisAIExecutionProvider"]
               )
end = time.perf_counter()
print(f"Session creation: {(end-start)*1000:.2f} ms")

print('\nINFO: Running inference with VitisAIExecutionProvider...')
start = time.perf_counter()
ofm = onnx_session.run(None, input_data)
end = time.perf_counter()
print(f"First run time: {(end-start)*1000:.2f} ms")

start = time.perf_counter()
ofm = onnx_session.run(None, input_data)
end = time.perf_counter()
print(f"Second run time: {(end-start)*1000:.2f} ms")

avg = 0
for i in range(0,10, 1):
    start = time.perf_counter()
    ofm = onnx_session.run(None, input_data)
    end = time.perf_counter()
    avg = avg + (end-start)
print(f"100 run avg: {(avg/100)*1000:.2f} ms")
print('INFO: Inference with VitisAIExecutionProvider completed')