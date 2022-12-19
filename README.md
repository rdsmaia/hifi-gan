# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

## Intro

This repo is a fork from https://github.com/jik876/hifi-gan, with very few modifications:

* It keeps the old format of the aforementioned repo.
* It is adjusted to read given mel spectrograms.

To understand HiFi-GAN read the original [paper](https://arxiv.org/abs/2010.05646).

## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)

## Training
```
python train.py \
  --config configs/config_v1.json \
  --input_wavs_dir dataset/ptBR/audio \
  --input_mels_dir dataset/ptBR/mels \
  --input_training_file dataset/train_files.txt \
  --input_validation_file dataset/test_files.txt
```
To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

## Pretrained Model
Here is a [pretrained model](https://drive.google.com/file/d/1iPNJMWKIUVmtA4hBOwyy7qs5M6EEvsa8/view?usp=share_link) trained on 20hs of a multispeaker ptBR dataset.

## Inference for end-to-end speech synthesis
1. Make `test_mel_dir` directory and copy generated mel-spectrogram files into the directory.<br>
The spectrograms produced by this model are compatible with the pretrained checkpoint, for instance:
[Tacotron2](https://github.com/rdsmaia/Tacotron-2).
2. Run the following command.
    ```
    python inference_e2e_from_folder.py --input_mels [test_mel_dir] --output_dir --output_dir [output_wav] --checkpoint_file [generator checkpoint file path] --npyin True
    ```
Generated wav files are saved in `output_wav`.<br>


## Acknowledgements
Many thanks to [Jungil Kong, Jaehyeon Kim, Jaekyoung Bae](https://github.com/jik876/hifi-gan) for making the original repo available.

