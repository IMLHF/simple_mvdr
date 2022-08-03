# author : red_wind@foxmail.com
# date   : 2022.07.26
# disc   : basic mvdr, torch gather speed


import math
import scipy
import soundfile as sf
from scipy import signal
import numpy as np
import torch
import librosa
from tqdm import tqdm


def mvdr(arraydata, name):
    '''
    arraydate: [L, M], L wav length, M mic number.
    '''
    d = 0.027 # array unit distance
    c = 340

    M = arraydata.shape[-1]

    sensor = np.array(list(range(0, 4)))

    win_len = 320
    frame_len = 320
    inc = 160

    # torch
    device = torch.device('cpu')
    # device = torch.device('cuda:0')
    sensor = torch.from_numpy(sensor).to(device)
    window = torch.kaiser_window(win_len, True, 6).to(device)
    wav = torch.from_numpy(arraydata).to(device) # [L, C]
    wav = wav - torch.mean(wav, dim=0, keepdim=True)
    wav = wav/wav.abs().max()

    # import scipy.io as scio
    # data=scio.loadmat('./fft_matlab.mat')
    # print(data.keys())
    # fft_cell_X = torch.from_numpy(data['fft_cell_X'])
    # print(fft_cell_X.shape, "data.shape")
    # print(fft_cell_X)

    fft_cell_X = torch.stft(wav.T, frame_len, inc, win_len, window, return_complex=True)
    # print(fft_cell_X.shape) #[ C, F, T]
    fft_cell_X = fft_cell_X.transpose_(0, 1) # [F, C, T]
    print(fft_cell_X.shape, "fft_cell_X.shape")

    n_frame = fft_cell_X.shape[-1]


    angle_s_s1 = -50.
    angle_s_s2 = 50.

    mvdr_beam_stft_s1 = torch.zeros([win_len//2+1, n_frame]).to(torch.complex128)
    mvdr_beam_stft_s2 = torch.zeros([win_len//2+1, n_frame]).to(torch.complex128)

    chwin = torch.ones([M, 1])

    f_k = np.array(list(range(1, win_len//2+1)))*fs//frame_len
    # print(f_k)

    RXX = torch.zeros([win_len//2+1, M, M], dtype=torch.complex128)
    for i in tqdm(list(range(0, n_frame)), ncols=50):
        for j in range(0, win_len//2-1):
            X = fft_cell_X[:, :, i].T # [C, F]
            temp = fft_cell_X[j+1, :, i:i+1] # [C, 1]
            a = 0.05
            b = 1.0 - a
            if i == 0:
                RXX[j,:,:] = temp @ temp.conj().T / temp.shape[-1]
            else:
                RXX[j,:,:] = a * RXX[j,:,:] + b * (temp @ temp.conj().T) / temp.shape[-1]

            aa = 1.
            invR = torch.linalg.inv(RXX[j,:,:] + aa*torch.trace(RXX[j,:,:])/M*torch.eye(M))
            # print("\n", invR)
            # exit(0)

            # s1
            h_d = torch.exp( -1j*2*math.pi*f_k[j]*sensor*d*np.sin(angle_s_s1*math.pi/180)/c
                             ).unsqueeze(1).conj().to(dtype=torch.complex128)
            w_mvdr = invR@h_d/(h_d.conj().T@invR@h_d)
            Beam_MVDR_temp = torch.multiply(chwin.T.conj(),w_mvdr.T.conj()) @ X[:,j+1:j+2]
            mvdr_beam_stft_s1[j+1,i] = Beam_MVDR_temp[0][0]

            # s2
            h_c = torch.exp( -1j*2*math.pi*f_k[j]*sensor*d*np.sin(angle_s_s2*math.pi/180)/c
                            ).unsqueeze(1).conj().to(dtype=torch.complex128)
            w_mvdr = invR@h_c/(h_c.conj().T@invR@h_c)
            Beam_MVDR_temp = torch.multiply(chwin.T.conj(),w_mvdr.T.conj()) @ X[:,j+1:j+2]
            mvdr_beam_stft_s2[j+1,i] = Beam_MVDR_temp[0][0]

    en_wav_s1 = torch.istft(mvdr_beam_stft_s1.unsqueeze(0), frame_len, inc, win_len)
    sf.write("py_mvdr_s1_%s.wav"%name, en_wav_s1[0], fs)
    en_wav_s2 = torch.istft(mvdr_beam_stft_s2.unsqueeze(0), frame_len, inc, win_len)
    sf.write("py_mvdr_s2_%s.wav"%name, en_wav_s2[0], fs)


if __name__ == "__main__":
    # s1, infs = sf.read("../beamforming_matlab/pos1-1272/128104/1272-128104-0000.flac")
    # s2, infs = sf.read("../beamforming_matlab/pos2-1673/143396/1673-143396-0000.flac")
    # min_len = min(s1.shape[0], s2.shape[0])
    # data = (s1[:min_len]+s2[:min_len])/2
    # sf.write("mix_sim_data.wav", data, infs)

    # data, infs = sf.read(r'./Multi_out_changjing6.wav')
    # data = data[:, :4]
    data, infs = sf.read(r'./multi_c_in.wav')
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
    mvdr(arraydata, "")

