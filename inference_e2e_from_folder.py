from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os, time
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write

from hifigan.env import AttrDict
from hifigan.meldataset import MAX_WAV_VALUE
from hifigan.models import Generator
from hifigan.utils import read_binfile, load_checkpoint

h = None
device = None


def inference(a):

    print('\nUsing device:', device)

    print('\nLoading model...')
    start_time = time.time()
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    load_time = time.time() - start_time
    message = f'Time spent: {load_time} sec\n'
    print(message)

    #filelist = os.listdir(a.input_mels_dir)
    filelist = [f for f in os.listdir(a.input_mels_dir) if f.endswith('.npy')]

    os.makedirs(a.output_dir, exist_ok=True)

    num_mels = h.num_mels
    sr = h.sampling_rate

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        accum_time = 0
        accum_dur = 0
        for i, filname in enumerate(filelist):

            start_time = time.time()

            if a.npyin is True or a.npyin == 'True':
                x = np.load(os.path.join(a.input_mels_dir, filname), allow_pickle=True).T
            else:
                x = read_binfile(os.path.join(a.input_mels_dir, filname), num_mels).T
            x = torch.FloatTensor(x).to(device)
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.wav')
            write(output_file, h.sampling_rate, audio)

            end_time = time.time() - start_time
            dur = len(audio) / sr

            total_time = load_time+end_time
            xrt = dur / total_time
            message = f'File: {filname} : Dur: {dur}, Syn time: {end_time} sec, Total (load+syn): {total_time} sec, xRT (dur/total): {xrt}'
            print(message)

            accum_dur += dur
            accum_time += end_time

        xrt = accum_dur / (accum_time + load_time)
        message = f'Toral dur: {accum_dur}, Syn time: {accum_time} sec, Total (load+syn): {load_time+accum_time} sec, xRT (dur/total): {xrt}\n'
        print(message)

        if device == torch.device('cuda'):
            print(torch.cuda.get_device_name(device=device))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(device=device)/1024**3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(device=device)/1024**3, 1), 'GB')
            print('Maximum memory allocated: ', round(torch.cuda.max_memory_allocated(device=device)/1024**3,1), 'GB\n')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--npyin', default=False)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

