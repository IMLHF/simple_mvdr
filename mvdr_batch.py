# author : red_wind@foxmail.com
# date   : 2022.07.26
# disc   : basic mvdr, torch gather speed


import math
from matplotlib.pyplot import axes
import scipy
import soundfile as sf
from scipy import signal
import numpy as np
import torch
import librosa
from tqdm import tqdm
from torch import nn

class MVDR(nn.Module):
    def __init__(self, array_dis=0.027, n_channel=4, s1_angle=-40., s2_angle=40.) -> None:
        super().__init__()
        self.array_dis = array_dis
        self.c = 340
        self.win_len = 320
        self.frame_len = 320
        self.inc = 160
        self.n_channel = n_channel
        self.s1_angle = s1_angle
        self.s2_angle = s2_angle

        sensor = torch.from_numpy(np.array(list(range(0, n_channel))))
        self.sensor = nn.Parameter(sensor, requires_grad=False)
        window = torch.kaiser_window(self.win_len, True, 6)
        self.window = nn.Parameter(window, requires_grad=False)
        chwin = torch.ones([1, self.n_channel, 1])
        self.chwin = nn.Parameter(chwin, requires_grad=False)
        f_k = np.array(list(range(0, self.win_len//2+1)))*fs//self.frame_len
        self.f_k = nn.Parameter(torch.from_numpy(f_k), requires_grad=False)

        # s1 steering_vec; s2 steering_vec; 
        self.h_s1 = torch.exp( (-1j*2*math.pi*self.f_k).unsqueeze(1)*self.sensor.unsqueeze(0)*self.array_dis*np.sin(
            self.s1_angle*math.pi/180)/self.c).unsqueeze(2).conj().to(dtype=torch.complex128).unsqueeze(0)  # [1, F, c, 1]
        self.h_s2 = torch.exp( (-1j*2*math.pi*self.f_k).unsqueeze(1)*self.sensor.unsqueeze(0)*self.array_dis*np.sin(
            self.s2_angle*math.pi/180)/self.c).unsqueeze(2).conj().to(dtype=torch.complex128).unsqueeze(0)  # [F, c, 1]
    
    def forward(self, mc_wav):
        '''
        mc_wav: [batch, L, C] L wav length, C mic number.
        '''
        batch = mc_wav.shape[0]
        mc_wav = mc_wav - torch.mean(mc_wav, dim=1, keepdim=True)
        mc_wav = mc_wav/mc_wav.abs().max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        bl_wav = mc_wav.transpose(-1, -2).view(batch*self.n_channel, -1) # [batch*C, L]
        fft_cell_X = torch.stft(bl_wav, self.frame_len, self.inc, 
                                self.win_len, self.window, return_complex=True)
        _, F, n_frame = fft_cell_X.shape
        fft_cell_X = fft_cell_X.reshape(batch, self.n_channel, F, n_frame)
        # print(fft_cell_X.shape) #[batch, C, F, T]
        fft_cell_X = fft_cell_X.transpose_(1, 2) # [batch, F, C, T]
        print(fft_cell_X.shape, "fft_cell_X.shape")

        mvdr_beam_stft_s1 = []
        mvdr_beam_stft_s2 = []

        RXX = torch.zeros([batch, self.win_len//2+1, self.n_channel, self.n_channel], dtype=torch.complex128)
        for i in tqdm(list(range(0, n_frame)), ncols=50):
            # for j in range(0, win_len//2):
            X = fft_cell_X[:, :, :, i] # [B, F, C]
            temp = fft_cell_X[:, :, :, i:i+1] # [B, F, C, 1]
            a = 0.05
            b = 1.0 - a
            if i == 0:
                RXX[:,:,:,:] = temp @ temp.conj().transpose(-1, -2) / temp.shape[-1]
            else:
                RXX[:,:,:,:] = a * RXX[:,:,:,:] + b * (temp @ temp.conj().transpose(-1, -2)) / temp.shape[-1]

            aa = 1.
            trace = (RXX[:,:,:,:] * torch.eye(self.n_channel).unsqueeze(0).unsqueeze(0)).sum([-2,-1], keepdim=True)
            invR = torch.linalg.inv(RXX[:,:,:,:] + aa*trace/self.n_channel*torch.eye(self.n_channel).unsqueeze(0).unsqueeze(0))
            # print("\n invR", invR.shape)
            # exit(0)

            # s1
            w_mvdr = invR@self.h_s1/(self.h_s1.conj().transpose(-1, -2)@invR@self.h_s1)
            # print(w_mvdr.shape, "w_mvdr.shape")
            Beam_MVDR_temp = torch.multiply(self.chwin.transpose(-1, -2).conj(),w_mvdr.transpose(-1, -2).conj()) @ X.unsqueeze(-1)
            # print(Beam_MVDR_temp.shape) # [batch, F, 1, 1]
            mvdr_beam_stft_s1.append(Beam_MVDR_temp[:,:, 0, 0]) 

            # s2
            w_mvdr = invR@self.h_s2/(self.h_s2.conj().transpose(-1, -2)@invR@self.h_s2)
            Beam_MVDR_temp = torch.multiply(self.chwin.transpose(-1, -2).conj(),w_mvdr.transpose(-1, -2).conj()) @ X.unsqueeze(-1)
            mvdr_beam_stft_s2.append(Beam_MVDR_temp[:,:, 0, 0])
        mvdr_beam_stft_s1 = torch.stack(mvdr_beam_stft_s1).permute(1, 2, 0) # [B, F, T]
        mvdr_beam_stft_s2 = torch.stack(mvdr_beam_stft_s2).permute(1, 2, 0)

        en_wav_s1 = torch.istft(mvdr_beam_stft_s1, self.frame_len, self.inc, self.win_len) # [B, L]
        # sf.write("py_mvdr_s1_%s.wav"%name, en_wav_s1[0], fs)
        en_wav_s2 = torch.istft(mvdr_beam_stft_s2, self.frame_len, self.inc, self.win_len) # [B, L]
        # sf.write("py_mvdr_s2_%s.wav"%name, en_wav_s2[0], fs)
        return en_wav_s1, en_wav_s2



if __name__ == "__main__":
    # s1, infs = sf.read("../beamforming_matlab/pos1-1272/128104/1272-128104-0000.flac")
    # s2, infs = sf.read("../beamforming_matlab/pos2-1673/143396/1673-143396-0000.flac")
    # min_len = min(s1.shape[0], s2.shape[0])
    # data = (s1[:min_len]+s2[:min_len])/2
    # sf.write("mix_sim_data.wav", data, infs)

    # data, infs = sf.read(r'./Multi_out_changjing6.wav')
    # data = data[:, :4]
    data, infs = sf.read(r'./multi_c_in.wav')
    # data, infs = sf.read(r'./t60_0.12_chinese_38_5721_20170914195214_38_5721_20170915090853.wav')
    print(data.shape, infs)
    fs = 16000

    if infs !=fs:
        lst = []
        for i in range(4):
            lst.append(librosa.resample(data[:, i], infs, fs))
        arraydata = np.array(lst).T
    else:
        arraydata = data
    print(arraydata.shape)
    # sf.write("multi_c_in.wav", arraydata, fs)
    # mvdr(arraydata, "batch")

    model = MVDR()
    s1_wav, s2_wav = model.forward(torch.from_numpy(arraydata).unsqueeze(0))
    sf.write("py_mvdr_s1_batch.wav", s1_wav[0].numpy(), fs)
    sf.write("py_mvdr_s2_batch.wav", s2_wav[0].numpy(), fs)

