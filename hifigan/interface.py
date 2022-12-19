from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import json
import torch
import numpy as np
import soundfile as sf

from .env import AttrDict
from .models import Generator
from .utils import read_binfile, load_checkpoint


def load_model(model_path):
    """Load model checkpoint and config
    Return generator and hparams
    """
    config_file = os.path.join(model_path, 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    hp = AttrDict(json_config)

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        hp.device = torch.device('cuda')
    else:
        hp.device = torch.device('cpu')

    print('\nUsing device:', hp.device)

    print('\nLoading model...')
    start_time = time.time()
    generator = Generator(hp).to(hp.device)
    checkpoint_file = os.path.join(model_path, 'best_netG.pt')
    state_dict_g = load_checkpoint(checkpoint_file, hp.device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    load_time = time.time() - start_time
    message = f'Time spent: {load_time} sec\n'
    print(message)

    return generator, hp


def synthesis(mel_file, wav_file, generator, hp):
    """Inverse mel to waveform
    Returns the duration of the output.
    """
    start_time = time.time()
    mel = read_binfile(mel_file, hp.num_mels).T
    mel = torch.FloatTensor(mel).to(hp.device)
    mel = mel.unsqueeze(0)  # add batch (size = 1)
    sr = hp.sampling_rate
    with torch.no_grad():
        audio = generator(mel).squeeze().cpu().numpy()
    torch.cuda.empty_cache()
    del generator
    sf.write(wav_file, audio / np.max(np.abs(audio)), sr, subtype="PCM_16")

    total_time = time.time() - start_time
    dur = len(audio) / sr

    xrt = dur / total_time
    message = f'File: {wav_file} : Dur: {dur}, Syn time: {total_time} sec, xRT (dur/total): {xrt}\n'
    print(message)

    return dur
