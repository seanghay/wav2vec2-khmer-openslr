#!/usr/bin/env python3
import numpy as np
import os
import json
import evaluate
from transformers import (
  Wav2Vec2ForCTC,
  Wav2Vec2Processor,
  Wav2Vec2CTCTokenizer,
  Wav2Vec2FeatureExtractor,
  TrainingArguments,
  Trainer,
)
from data import create_dataset
from data_collate import DataCollatorCTCWithPadding
import click


@click.command()
@click.option("--audio_dir", help="Audio directory", default="km_kh_male/wavs")
@click.option(
  "--metadata_file", help="Metadata file", default="km_kh_male/line_index.tsv"
)
@click.option("--model_id", help="Model ID", default="facebook/wav2vec2-base")
@click.option("--output_dir", help="Output dir", default="result")
@click.option("--model_path", help="Model output path", default="model")
@click.option("--batch_size", type=click.INT, help="Per device batch size", default=32)
@click.option("--epoch", type=click.INT, default=30)
def train(audio_dir, metadata_file, model_id, output_dir, batch_size, epoch, model_path):
  audio_dataset, vocab_dict = create_dataset(
    metadata_file=metadata_file, audio_dir=audio_dir
  )

  if not os.path.exists("vocab.json"):
    with open("vocab.json", "w") as outfile:
      json.dump(vocab_dict, outfile, ensure_ascii=False, indent=2)

  wer_metric = evaluate.load("wer")
  tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    clean_up_tokenization_spaces=False,
  )

  feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
  )

  processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
  )

  def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

  def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(
      audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
      batch["labels"] = processor(batch["text"]).input_ids
    return batch

  data = audio_dataset.map(prepare_dataset, remove_columns=audio_dataset.column_names)
  data = data.train_test_split(test_size=0.1, shuffle=False, seed=42)
  data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

  model = Wav2Vec2ForCTC.from_pretrained(
    model_id,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
  )

  model.freeze_feature_encoder()

  training_args = TrainingArguments(
    output_dir=output_dir,
    report_to="tensorboard",
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    eval_strategy="steps",
    num_train_epochs=epoch,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=100,
    save_total_limit=5,
    push_to_hub=False,
    load_best_model_at_end=True,
  )

  trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=processor.feature_extractor,
  )

  trainer.train()

  # save model when finished
  model.save_pretrained(model_path)
  processor.save_pretrained(model_path)


if __name__ == "__main__":
  train()
