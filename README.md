<div align="center">
<samp>
  
<h1> EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion </h1>

<h3> <b>AAAI 2026</h3>

</samp>


</div>     

## Environment Setup Guidance
```Install step
conda create -n EndoIR python=3.10
conda activate EndoIR
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../EndoIR
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
```

## Test
```
python endoir/train.py -opt configs/test.yaml
```
## Train

python endoir/train.py -opt configs/train.yaml


## Thanks
```
This repository is built on BasicSR, thanks to their open-source contributions.
```
