#!/usr/bin/env python

# @Author: Richard Yue

import argparse

parser = argparse.ArgumentParser(
    prog="T5 trainer script",
    description="Trains T5 on a dataset",
    epilog="Use at your own risk!"
)

parser.add_argument('-m', '--model', help="The model checkpoint to train", type=str)
parser.add_argument('-d', '--dataset', help="The dataset to train on", type=str)
parser.add_argument('-e', '--epochs', help="The epoch number to append to the end of the filename", type=int)
parser.add_argument('-lr', '--learning_rate', help="The learning rate for training", type=float)

args = parser.parse_args()

checkpoint = args.model if args.model else "t5-small"
from datasets import load_dataset
if args.dataset:
    dataset = load_dataset(args.dataset)
else:
    dataset = load_dataset("Helsinki-NLP/europarl", "en-fr")
epochs = int(args.epochs) if args.epochs else 1
lr = float(args.learning_rate) if args.learning_rate else 2e-05

# %%
raw_dataset_train_test = dataset["train"].train_test_split(test_size=0.3, seed=42)
raw_dataset_test_val = raw_dataset_train_test["test"].train_test_split(test_size=0.00975126156, seed=42)

# %%
from transformers import AutoTokenizer, set_seed

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
set_seed(42)

# %%
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [prefix + example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    labels = tokenizer(text_target=targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
tokenized_train = raw_dataset_train_test['train'].map(preprocess_function, batched=True)
# tokenized_test = raw_dataset_test_val['train'].map(preprocess_function, batched=True)
# tokenized_val = raw_dataset_test_val['test'].map(preprocess_function, batched=True)

# %%
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from peft import LoraConfig, TaskType, PeftModel, PeftConfig, get_peft_model

if checkpoint == "t5-small":
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = get_peft_model(model, peft_config)

else:
    config = PeftConfig.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(
        model,
        checkpoint,
        is_trainable=True)
    model.set_adapter('default')
    model.print_trainable_parameters()
    
model.to(device)
print(f"Device: {device}")

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# %%
import evaluate
from tqdm.auto import tqdm
bleu = evaluate.load("bleu")

def translate(text):
    input_ids = tokenizer(f"translate English to French: {text}", return_tensors="pt", truncation=True)["input_ids"].to(device)
    outputs = model.generate(input_ids)[0]
    return tokenizer.decode(outputs, skip_special_tokens=True)[29:]

def get_label(example, lang):
    return {"references":example["translation"][lang]}

def compute_metrics(predictions, references):
    results = bleu.compute(predictions=predictions, references=references)
    return results

# %%
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='no',
    learning_rate=lr,
    per_device_train_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# %%

model.save_pretrained(f"./t5_finetuned_europarl_epoch_{epochs}")
tokenizer.save_pretrained(f"./t5_finetuned_europarl_epoch_{epochs}")

torch.backends.cuda.matmul.allow_tf32 = True

model.eval()

# %%
fr_labels = raw_dataset_test_val.map(get_label, fn_kwargs={'lang':'fr'})
en_labels = raw_dataset_test_val.map(get_label, fn_kwargs={'lang':'en'})
predictions = []
fr_references = []
en_references = []
for label in fr_labels["test"]:
    fr_references.append(label["references"])
for label in en_labels["test"]:
    en_references.append(label["references"])
for i in tqdm(range(len(raw_dataset_test_val["test"]))):
    predictions.append(translate(raw_dataset_test_val["test"]["translation"][i]["en"]))
print("BLEU:")
print(compute_metrics(predictions=predictions, references=fr_references))

# %%
comet_metric = evaluate.load('comet')
source = en_references
comet_score = comet_metric.compute(predictions=predictions, references=fr_references, sources=source)
print("Comet:")
print(comet_score['mean_score'])

# %%

bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=predictions, references=fr_references, model_type="distilbert-base-uncased")
print("Bertscore:")

def get_average(l:list):
    return (sum(l)/len(l))

avg_precision = get_average(results["precision"])
avg_recall = get_average(results["recall"])
avg_f1 = get_average(results["f1"])
print({"average_precision":avg_precision, "average_recall":avg_recall, "average_f1":avg_f1})

meteor = evaluate.load("meteor")
results = meteor.compute(predictions=predictions, references=fr_references)
print("Meteor:")
print(results)
