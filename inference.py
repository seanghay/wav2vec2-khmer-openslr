import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

if __name__ == "__main__":
  model = Wav2Vec2ForCTC.from_pretrained("./model")
  processor = Wav2Vec2Processor.from_pretrained("./model")

  y, sr = librosa.load("samples/khm_1161_1980987674.wav", mono=True, sr=16000)
  values = processor(y, sampling_rate=sr, return_tensors="pt")

  with torch.no_grad():
    outputs = model(**values).logits
    text = processor.batch_decode(outputs.argmax(dim=-1))[0]
    print(text)
    # ប្អូន ប្រុស របស់ ខ្ញុំ បាន ទៅ លេង នៅ ទីក្រុង ម៉ានីល
