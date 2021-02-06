import numpy as np
import torchaudio
import torch.nn as nn
import torch
        
class AudioPreprocessor(): 
    
    def __init__(self):
        """ Initializes MelSpectrogram function. """
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=480,
            hop_length=160,
            f_min=0,
            f_max=8000,
            n_mels=40,
            window_fn=torch.hann_window
        )
    
    def amplitude_to_db(self, data):
        """ Converts spectrogram to logarithmic scale.

            Args:
                Torch tensor (Mel Spectrogram). Dimensions are: 1 x coefficients x time.
            Returns: 
                Torch Tensor (log-Mel Spectrogram).
        """
        data[data > 0] = torch.log(data[data > 0])
        # return data.view(data.shape[0], data.shape[1], -1)
        return data

    def load_audio(self, audio_file):
        """ Loads audio wav file into torch tensor.

            Args:
                audio_file (string).
            Returns:
                Torch tensor of shape [1 x 16000] (audio files are sampled at 16 kHz).
        """
        return torchaudio.load_wav("../data/" + audio_file + ".wav")[0]

    def compute_log_mel_spectrogram(self, data):
        """ Computes log_mel_spectrogram.

            Args:
                Torch tensor of shape [1 x 16000] representing audio file.
            Returns:
                Torch tensor of shape [1 x n_mels x 101] representing log Mel spectrogram. 
        """
        return self.amplitude_to_db(self.spectrogram(data))

    def compute_binary_spectrogram(self, data, threshold=12):
        """ Computes power variation spectrogram.

            Args:
                Torch tensor of shape [1 x n_mels x 101] representing log Mel spectrogram. 
            Returns:
                Torch tensor of shape [1 x n_mels x 101] representing the power variation spectrogram. 
        """
        # Rescaling to 0-255
        delta = data.max() - 20.0
        data = data - delta
        data[data<0] = 0
        data = data*12.75
        # Computing variation of power
        ref_val = data[0,:,0]
        output = torch.zeros((data.shape[0], data.shape[1], data.shape[2]))
        one_mask = torch.ones(data.shape[1])
        for x in range(data.shape[2]-1):
            temp_val = data[0,:,x+1] - ref_val
            d_up = 1.0*torch.ge(temp_val, threshold)
            d_down = 1.0*torch.le(temp_val, -threshold)
            # Update of reference
            ref_val = (one_mask-(d_up+d_down))*ref_val + (d_up+d_down)*data[0,:,x+1]
            # Creation of new vector
            output[:,:,x] = (d_up - d_down)  

        return output

if __name__ == "__main__":
    pass

