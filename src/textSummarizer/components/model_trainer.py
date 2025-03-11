from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from transformers import TrainingArguments,Trainer
from transformers import DataCollatorForSeq2Seq
import torch
import os
from src.textSummarizer.entity import ModelTrainerConfig
from datasets import load_from_disk

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import TrainingArguments
class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config
    
    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        ##loading the data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps, evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps, save_steps=1e6, gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        trainer = Trainer(
            model=model_pegasus,
            args= trainer_args,
            tokenizer=tokenizer,
            train_dataset=dataset_samsum_pt['test'], # training on test as it is smaller
            eval_dataset=dataset_samsum_pt['validation'],
            data_collator=seq2seq_data_collator
        )

        trainer.train()

        ## save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

