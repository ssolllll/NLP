from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

import numpy as np

class TranslationTrain:
    def __init__(self,checkpoint):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        pass

    def save_model(self):
        self.trainer.save_model('../translation/0129')
        return None

    def train(self) -> object :
        self.trainer.train()
        return None 

    def _postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def _compute_metrics(self,eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def _set_arguments(self, train,test):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model = self.checkpoint)
        self.metric = evaluate.load("sacrebleu")

        training_args = Seq2SeqTrainingArguments(
            output_dir="translation_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=10,
            predict_with_generate=True,
            # fp16=True,
            eval_steps=500,
            save_steps=500,
            push_to_hub=True
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


tokenizer = AutoTokenizer.from_pretrained('t5-small')
def _preprocessing_function(examples,
                                source_lang = "en",target_lang = "fr",
                                prefix = f"translate English to French: "):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

if __name__ == "__main__":
    from datasets import load_dataset
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)
    tokenized_dataset = books.map(_preprocessing_function, batched=True)
    print(tokenized_dataset)

    trainer = TranslationTrain(checkpoint = 't5-small')
    trainer._set_arguments(train=tokenized_dataset['train'],
                           test = tokenized_dataset['test'])
    trainer.train()
