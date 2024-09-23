## Wav2Vec2 with OpenSLR 42 (Khmer language)

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


