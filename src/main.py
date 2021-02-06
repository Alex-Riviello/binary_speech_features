from AudioPreprocessor import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

AUDIO_LIST = ["left", "right", "up", "down", "on", "off", "go", "stop", "yes", "no"]

if __name__ == "__main__":
    ap = AudioPreprocessor()

    fig = plt.figure()
    outer_grid = gridspec.GridSpec(5, 2, wspace=0.4, hspace=0.2)

    for i, audio_file in enumerate(AUDIO_LIST):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i], wspace=0.1, hspace=0.1)

        # Loading audio and computing log Mel spectrogram
        data = ap.load_audio(audio_file)
        data = ap.compute_log_mel_spectrogram(data)

        ax1 = plt.Subplot(fig, inner_grid[0])
        ax1.imshow(data[0])
        ax1.axis('off')
        ax1.set_title(audio_file)
        fig.add_subplot(ax1)

        # Computing power-variation spectrogram
        data = ap.compute_binary_spectrogram(data).numpy()
        ax2 = plt.Subplot(fig, inner_grid[1])
        ax2.imshow(data[0])
        ax2.axis('off')
        fig.add_subplot(ax2)

    fig.show()



    
 