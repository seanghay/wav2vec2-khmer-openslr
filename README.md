## Wav2Vec2 with OpenSLR 42 (Khmer language)

```shell
apt update -y
apt install -y unzip tmux neovim
```

```shell
conda create -n w2v2 python=3.8 --yes
conda activate w2v2

# install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --yes
```

### Dependencies

```shell
pip install -r requirements.txt
```

### Training

```
./prepare.sh

./train.py
```

### Inference

Download the weights via: [GitHub Release](https://github.com/seanghay/wav2vec2-khmer-openslr/releases/tag/v1.0.0)

```python
python inference.py samples/audio.wav

# => វិធាណការ៍ ដាប បំរាំ គួរចរ ក្នុង រាជធានី ភ្នំពេញដើម បី ស្វត្ឋិភាព សហគម បាន អនប័តរ យ៉ាង រ្លូន
```