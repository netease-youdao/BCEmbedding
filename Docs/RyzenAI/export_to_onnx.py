#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
import numpy as np

def process_items(item_string):
    # Split the string into individual items and convert it to a list
    item_list = item_string.split(',')
    return item_list

def prepare_text_input(tokenizer, token_length):
    """
    Prepare dummy input for text models
    """
    return tokenizer("Hello, world!", return_tensors="pt", padding='max_length', max_length=token_length)

def load_model(model_name: str):
    """
    Load model based on its type (timm or huggingface)
    """
    model = AutoModel.from_pretrained(model_name)
    processor = AutoTokenizer.from_pretrained(model_name)
            
    return model, processor

def export_model_to_onnx(model_name: str, output_dir: str, token_length, opset_version, model_inputs, model_outputs) -> str:
    """
    Export a Hugging Face PyTorch model to ONNX format.
    
    Args:
        model_name (str): Name or path of the Hugging Face model
        output_dir (str): Directory to save the ONNX model
        opset_version (int): ONNX opset version to use
        
    Returns:
        str: Path to the saved ONNX model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_name}")
    model, processor = load_model(model_name)
    
    # Prepare input based on model type
    dummy_input = prepare_text_input(processor, token_length)

    #Processing dummy inputs based on model requirement
    input_list = process_items(model_inputs)

    if len(input_list) == 1:
        input_names = ['input_ids']
        inputs = (dummy_input['input_ids'])

    elif len(input_list) == 3:
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        inputs = (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids'])
        
    else:
        input_names = ['input_ids', 'attention_mask']
        inputs = (dummy_input['input_ids'], dummy_input['attention_mask'])

    output_list = process_items(model_outputs)
    output_names=output_list

    # Set model to evaluation mode
    #model.eval()
    
    # Prepare output path
    model_name_safe = os.path.basename(model_name)
    output_path = os.path.join(output_dir, f"{model_name_safe}.onnx")
    
    # Export the model
    print(f"Exporting model to ONNX format (opset version: {opset_version})")
    with torch.no_grad():
        print("Configuring export for text model")
        
        torch.onnx.export(
            model,                     # PyTorch model
            inputs,                    # model input
            output_path,              # output path
            opset_version=opset_version,
            input_names=input_names,  # model input names
            output_names=output_names   # model output names
           
        )
    print(f"Model exported successfully to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Export Hugging Face PyTorch model to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        help='Name or path of the Hugging Face model')
    parser.add_argument('--token_length', type=int, default=256,
                        help='ONNX opset version to use')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the ONNX model')
    parser.add_argument('--opset', type=int, default=19,
                        help='ONNX opset version to use')
    parser.add_argument('--model_inputs', type=str, help='Comma-separated items (e.g., "input_ids,attention_mask,token_type_ids")')
    parser.add_argument('--model_outputs', type=str, help='Comma-separated items (e.g., "start_logits,end_logits")')
    
    args = parser.parse_args()
    
    output_path = export_model_to_onnx(
        model_name=args.model,
        output_dir=args.output_dir,
        token_length=args.token_length,
        opset_version=args.opset,
        model_inputs=args.model_inputs,
        model_outputs=args.model_outputs
    )
    
    print(f"ONNX model path: {output_path}")

if __name__ == "__main__":
    main() 