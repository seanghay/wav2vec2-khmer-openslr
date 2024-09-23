import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import click


@click.command()
@click.argument("file")
def cli(file):
  model = Wav2Vec2ForCTC.from_pretrained("./model")
  processor = Wav2Vec2Processor.from_pretrained("./model")

  y, sr = librosa.load(file, mono=True, sr=16000)
  values = processor(y, sampling_rate=sr, return_tensors="pt")

  with torch.no_grad():
    outputs = model(**values).logits
    text = processor.batch_decode(outputs.argmax(dim=-1))[0]
    print(text)


if __name__ == "__main__":
  cli()
