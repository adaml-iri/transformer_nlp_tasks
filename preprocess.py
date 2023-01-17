import logging
import argparse
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, TFAutoModelForTokenClassification
import pandas as pd
import os
import torch
import pysbd
from itertools import islice


class NLP_Preprocessing:
    model_name = None
    model_type = None
    auto_model = None
    tokenizer_name = None
    annotate_tokenizer = None
    initialized = False
    infile = None
    output = None
    lang = None
    device = 0 if torch.cuda.is_available() else -1

    def __init__(self, model_name, tokenizer_name, model_type, infile, output, lang):
        if model_name is not None and len(model_name) > 0:
            self.model_name = str(model_name)
        if tokenizer_name is not None and len(tokenizer_name) > 0:
            self.tokenizer_name = str(tokenizer_name)
        if model_type is not None and len(model_type) > 0:
            self.model_type = str(model_type).upper()
        
        if infile is not None:
            self.infile = str(infile)
        if output is not None:
            self.output = str(output)
        if lang is not None:
            self.lang = str(lang)
        if not self.initialized:
            self.initialized = True

    def preprocess(self):
        tab = '\t'
        newline = '\n'
        model = None
        if self.model_type == "PYTORCH":
            model = nlp.pytorch_model()
        elif self.model_type == "TENSORFLOW":
            model = nlp.tensorflow_model()
        else :
            return -1
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, aggregation_strategy="none")
        seg = pysbd.Segmenter(language=self.lang, clean=False)
        N = 100
        count = 0
        text = ""
        out_file = open(self.output, "a")
        in_file = open(self.infile, 'r')
        line = in_file.readline()
        while line != "":
            text = text + line
            count +=1
            if (count == N) :
                count = 0
                list_of_sentences = seg.segment(text)
                for sentence in list_of_sentences:
                    sentence = sentence.strip()       
                    inputs = tokenizer(sentence, return_tensors="pt")
                    tokens = inputs.tokens()
                    outputs = model(**inputs).logits
                    predictions = torch.argmax(outputs, dim=2)
                    for token, prediction in zip(tokens, predictions[0].numpy()):
                        if (token == "[SEP]" or token == "[CLS]"):
                            continue
                        line_in_csv = token + tab +model.config.id2label[prediction] + newline
                        out_file.write(line_in_csv)
                    
                text = ""
            line = in_file.readline()
        else :
            # Handle remaing lines
            list_of_sentences = seg.segment(text)
            for sentence in list_of_sentences:
                sentence = sentence.strip()       
                inputs = tokenizer(sentence, return_tensors="pt")
                tokens = inputs.tokens()
                outputs = model(**inputs).logits
                predictions = torch.argmax(outputs, dim=2)
                for token, prediction in zip(tokens, predictions[0].numpy()):
                    if (token == "[SEP]" or token == "[CLS]"):
                        continue
                    line_in_csv = token + tab +model.config.id2label[prediction] + newline
                    out_file.write(line_in_csv)

            in_file.close()
            out_file.close()



    def pytorch_model(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        return model

    def tensorflow_model(self):
        model = TFAutoModelForTokenClassification.from_pretrained(self.model_name)
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to train huggingface transformer model with a custom dataset.")
    parser.add_argument('model_name', type=str, help="The name of the model or full path to model.")
    parser.add_argument('tokenizer_name', type=str, help="The name of tokenizer or full path to tokenizer.")
    parser.add_argument('model_type', type=str, choices=["PYTORCH", "TENSORFLOW", "pytorch", "tensorflow"],help="Provide model type (i.e. PYTORCH or TENSORFLOW")
    parser.add_argument('infile', type=str, help="Provide raw text file or full path to raw text file.")
    parser.add_argument('output_dir', type=str, help="Provide full path to directory where new trained model will placed.")
    parser.add_argument("language", type=str, help="Provide language model should detect with.")
    args = parser.parse_args()
    ready = True
    array = []
    model_name = args.model_name
    array[0] = model_name
    model_type = args.model_type
    array[1] = model_type
    tokenizer_name = args.tokenizer_name
    array[2] = tokenizer_name
    infile = args.infile
    array[3] = infile
    output_dir = args.output_dir
    array[4] = output_dir
    lang = args.language
    array[5] = lang
    for val in array:
        if val == "" or val == None:
            print("Need to provide all parameters.")
            ready = False
    ready = True
    if ready:
        nlp = NLP_Preprocessing(model_name, tokenizer_name, model_type, infile, output_dir, lang)
        success = nlp.preprocess()
        if success == -1:
            print("Failed to process text file.")
        else :
            print("Successfully processed text file.")