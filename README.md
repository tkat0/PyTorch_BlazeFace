# PyTorch_BlazeFace

[![Build Status](https://travis-ci.com/tkat0/PyTorch_BlazeFace.svg?branch=master)](https://travis-ci.com/tkat0/PyTorch_BlazeFace)

Unofficial PyTorch implementation of [BlazeFace](https://sites.google.com/view/perception-cv4arvr/blazeface)

## Installation

```bash
pip install git+https://github.com/tkat0/PyTorch_BlazeFace
```

```python
import torch
from blazeface import BlazeFace

x = torch.randn(1, 3, 128, 128)
model = BlazeFace()
h = model(x)
```

## Paper
### BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs
[[Project Page]](https://sites.google.com/view/perception-cv4arvr/blazeface)
[[Original Implementation]](https://github.com/google/mediapipe/tree/master/mediapipe/models#blazeface-face-detection-model)
