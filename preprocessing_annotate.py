import logging
import argparse
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, TFAutoModelForTokenClassification
import pandas as pd
import os
import torch

class NLP_Preprocessing_Annotate:
    model_name = None
    model_type = None
    auto_model = None
    tokenizer_name = None
    annotate_tokenizer = None
    initialized = False
    aggregation_strategy = None
    dataset = None
    output = None
    device = 0 if torch.cuda.is_available() else -1

    def __init__(self, model_name, tokenizer_name, model_type, aggregation_strategy, dataset, output):
        if model_name is not None and len(model_name) > 0:
            self.model_name = str(model_name)
        if tokenizer_name is not None and len(tokenizer_name) > 0:
            self.tokenizer_name = str(tokenizer_name)
        if model_type is not None and len(model_type) > 0:
            self.model_type = str(model_type).upper()
        if aggregation_strategy is not None and len(aggregation_strategy) > 0:
            self.aggregation_strategy = str(aggregation_strategy)
        if dataset is not None:
            self.dataset = str(dataset)
        if output is not None:
            self.output = str(output)
        if not self.initialized:
            self.initialized = True

    def annotate(self):
        tab = '\t'
        newline = '\n'
        model = None
        if self.model_type == "PYTORCH":
            model = nlp.pytorch_model(self)
        elif self.model_type == "TENSORFLOW":
            model = nlp.tensorflow_model(self)
        else :
            return -1
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, aggregation_strategy=self.aggregation_strategy)
        in_file = open(str(self.dataset), 'r')
        out_file = open(str(self.output), "a")
        line = in_file.readline()
        text = ""
        counter = 0
        while line != "":
            text = line.strip()
            if text == "":
                line = in_file.readline()
                continue
            else :
                line = in_file.readline()
                counter+=1
            inputs = tokenizer(text, return_tensors="pt")
            tokens = inputs.tokens()
            outputs = model(**inputs).logits
            predictions = torch.argmax(outputs, dim=2)
            for token, prediction in zip(tokens, predictions[0].numpy()):
                if (token == "[SEP]" or token == "[CLS]"):
                    continue
                line_in_csv = token + tab +model.config.id2label[prediction] + newline
                out_file.write(line_in_csv)
            out_file.write(newline)
        else :
            in_file.close()
            out_file.close()
        return 0

    def pytorch_model(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        return model

    def tensorflow_model(self):
        model = TFAutoModelForTokenClassification.from_pretrained(self.model_name)
        return model

nlp = NLP_Preprocessing_Annotate("dbmdz/bert-large-cased-finetuned-conll03-english", "bert-base-cased", "PYTORCH", "simple", "dataset.txt", "output.csv")
nlp.annotate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to train huggingface transformer model with a custom dataset.")
    parser.add_argument('model_name', type=str, help="The name of the model or full path to model.")
    parser.add_argument('tokenizer_name', type=str, help="The name of tokenizer or full path to tokenizer.")
    parser.add_argument('model_type', type=str, choices=["PYTORCH", "TENSORFLOW", "pytorch", "tensorflow"],help="Provide model type (i.e. PYTORCH or TENSORFLOW")
    parser.add_argument('dataset', type=str, help="Provide train dataset or full path to train dataset.")
    parser.add_argument('output_dir', type=str, help="Provide full path to directory where new trained model will placed.")
    args = parser.parse_args()
    ready = True
    array = []
    model_name = args.model_name
    array[0] = model_name
    model_type = args.model_type
    array[1] = model_type
    tokenizer_name = args.tokenizer_name
    array[2] = tokenizer_name
    dataset = args.dataset
    array[3] = dataset
    output_dir = args.output_dir
    array[4] = output_dir
    for val in array:
        if val == "" or val == None:
            print("Need to provide all parameters.")
            ready = False

    if ready:
        nlp = NLP_Preprocessing_Annotate(model_name, tokenizer_name, model_type, "simple", dataset, output_dir)
        success = nlp.annotate()
        if success == -1:
            print("Failed to train model.")
        else :
            print("Successfully trained model.")