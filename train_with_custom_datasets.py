from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, TFAutoModelForTokenClassification, AutoConfig
import numpy as np
import torch
import tensorflow as tf
import argparse

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NER_Trainer:
    model_name = None
    model_type = None
    tokenizer_name = None
    infile_dataset = None
    output_dir = None
    initialized = False


    def __init__(self, model_name, tokenizer_name, model_type, infile_dataset, output_dir):
        if model_name is not None:
            self.model_name = str(model_name)
        if model_type is not None:
            self.model_type = str(model_type).upper()
        if tokenizer_name is not None:
            self.tokenizer_name = str(tokenizer_name)
        if infile_dataset is not None:
            self.infile_dataset = str(infile_dataset)
        if output_dir is not None:
            self.output_dir = str(output_dir)
        if not self.initialized:
            self.initialized = True
       

    def read_wnut(in_file):
        file_path = Path(in_file)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
        return token_docs, tag_docs

    def encode_tags(tags, encodings, tag2id):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)
            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels

    def trainer(self):
        texts, tags = NER_Trainer.read_wnut(self.infile_dataset)
        train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)
        temp = AutoModelForTokenClassification.from_pretrained(self.model_name)
        
        id2tag = temp.config.id2label
        tag2id = temp.config.label2id
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        train_labels = NER_Trainer.encode_tags(train_tags, train_encodings, tag2id)
        val_labels = NER_Trainer.encode_tags(val_tags, val_encodings, tag2id)
        train_encodings.pop("offset_mapping") # we don't want to pass this to the model
        val_encodings.pop("offset_mapping")
        config = AutoConfig.from_pretrained(self.model_name, label2id=tag2id, id2label=id2tag)
        if (self.model_type == "PYTORCH") :
            train_dataset = WNUTDataset(train_encodings, train_labels)
            val_dataset = WNUTDataset(val_encodings, val_labels)
            NER_Trainer.pytorch_trainer(self, train_dataset, val_dataset,  config)
        elif (self.model_type == "TENSORFLOW"):
            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_encodings),
                train_labels
            ))
            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_encodings),
                val_labels
            ))
            NER_Trainer.tensorflow_trainer(self, train_dataset, val_dataset,  config)
        else:
            return -1
        return 0
        

    def pytorch_trainer(self, train_dataset, val_dataset, config):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=config)
        training_args = TrainingArguments(
            output_dir=self.output_dir,          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            report_to="none",
            save_strategy="no"
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )
        trainer.train()
        trainer.save_model(self.output_dir)

    def tensorflow_trainer(self, train_dataset, val_dataset, config):

        model = TFAutoModelForTokenClassification.from_pretrained(self.model_name, config=config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=model.hf_compute_loss) # can also use any keras loss fn
        model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
        model.save_pretrained(self.output_dir)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to train huggingface transformer model with a custom dataset.")
    parser.add_argument('model_name', type=str, help="The name of the model or full path to model.")
    parser.add_argument('tokenizer_name', type=str, help="The name of tokenizer or full path to tokenizer.")
    parser.add_argument('model_type', type=str, choices=["PYTORCH", "TENSORFLOW", "pytorch", "tensorflow"],help="Provide model type (i.e. PYTORCH or TENSORFLOW")
    parser.add_argument('train_dataset', type=str, help="Provide train dataset or full path to train dataset.")
    parser.add_argument('output_dir', type=str, help="Provide full path to directory where new trained model will placed.")
    args = parser.parse_args()
    ready = True
    array = []
    model_name = args.model_name
    model_type = args.model_type
    tokenizer_name = args.tokenizer_name
    train_dataset = args.train_dataset
    output_dir = args.output_dir
    array.insert(0, model_name)
    array.insert(1, model_type)
    array.insert(2, tokenizer_name)
    array.insert(3, train_dataset)
    array.insert(4, output_dir)
    for val in array:
        if val == "" or val == None:
            print("Need to provide all parameters.")
            ready = False
    
    if ready:
        ner = NER_Trainer(model_name, tokenizer_name, model_type, train_dataset, output_dir)
        success = ner.trainer()
        if success == -1:
            print("Failed to train model.")
        else :
            print("Successfully trained model.")
