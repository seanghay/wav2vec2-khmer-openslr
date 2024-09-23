import csv
import os
from datasets import Dataset, Audio


def create_dataset(metadata_file, audio_dir):
  with open(metadata_file) as infile:
    reader = csv.reader(infile, delimiter="\t")

    values = {
      "audio": [],
      "text": [],
    }

    all_text = ""

    for file_name, _, text in reader:
      audio_path = os.path.join(audio_dir, file_name + ".wav")
      values["audio"].append(audio_path)
      values["text"].append(text)
      all_text += text

    vocab_dict = {token: idx for idx, token in enumerate(sorted(set(all_text)))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    audio_dataset = Dataset.from_dict(values).cast_column(
      "audio", Audio(sampling_rate=16000, mono=True)
    )

    return audio_dataset, vocab_dict
