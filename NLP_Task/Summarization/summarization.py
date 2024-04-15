from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate
import torch

class SummarizationTrain:
    def __init__(self,checkpoint):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        pass

    def save_model(self):
        self.trainer.save_model('../summarization/0130')
        return None

    def train(self) -> object :
        self.trainer.train()
        return None 

    def _compute_metrics(self,eval_pred):
        predictions, labㄴels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def _set_arguments(self, train,test):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model = self.checkpoint)
        self.metric = evaluate.load("rouge")

        training_args = Seq2SeqTrainingArguments(
            output_dir="summarization_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
        )
    
        self.trainer = Seq2SeqTrainer(
            model = self.model,
            args = training_args,
            train_dataset = train,
            eval_dataset = test,
            tokenizer = self.tokenizer,
            data_collator = data_collator,
            compute_metrics = self._compute_metrics,
        )
        return None

class SummarizationInference:
    def __init__(self,
                 text : list,
                 checkpoint='t5-small'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.text = text
    
    def inference(self):
        outputs = []
        for idx in range(len(self.text)):
            inputs = self.tokenizer(self.text[idx], return_tensors="pt").input_ids
            with torch.no_grad():
                output = self.model.generate(inputs, max_new_tokens=100, do_sample=False)
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(summary)
        return outputs


tokenizer = AutoTokenizer.from_pretrained('t5-small')
def _preprocessing_function(examples,
                            prefix = "summarize: "):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    from datasets import load_dataset
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2)
    tokenized_dataset = billsum.map(_preprocessing_function, batched=True)
    print(tokenized_dataset)

    trainer = SummarizationTrain(checkpoint = 't5-small')
    trainer._set_arguments(train=tokenized_dataset['train'],
                           test = tokenized_dataset['test'])
    trainer.train()
