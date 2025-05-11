import numpy as np
from framework import dft_utils

#definindo senoide arbitrária para testar a DFT iterativa
f = 60
A = 2
init = 0
end = 2
n_samples = 1000
Ts = (end-init)/n_samples #período de amostragem
t = np.arange(init, end, Ts) #vetor de tempos do sinal
fs = 1/Ts #frequência de amostragem
delta_f = fs/n_samples #resolução do espectro
esc_freq = np.arange(-fs/2, fs/2, delta_f) #escala de frequências
sig_r = 1+A*np.sin(2*np.pi*f*t) + (A/2)*np.cos(2*np.pi*2*f*t) #parte real do sinal
sig_i = np.zeros_like(sig_r) #parte imaginária do sinal

#Rodando a DFT iterativa
DFT_r_it, DFT_i_it = dft_utils.iterDFT(sig_i, sig_r)
DFT_it = np.sqrt(DFT_r_it**2 + DFT_i_it**2)/n_samples #módulo

#Rodando a DFT parcialmente vetorizada
DFT_r, DFT_i = dft_utils.vectDFT(sig_i, sig_r)
DFT_vect = np.sqrt(DFT_r**2 + DFT_i**2)/n_samples #módulo

#Rodando a DFT vetorizada
optDFT_r, optDFT_i = dft_utils.optDFT(sig_i, sig_r)
optDFT_vect = np.sqrt(optDFT_r**2 + optDFT_i**2)/n_samples #módulo

#Rodando FFT numpy
DFT_np = np.fft.fft(sig_r+sig_i) #computa FFT
DFT_np = np.fft.fftshift(DFT_np) #shift na FFT
DFT_np = np.abs(DFT_np)/n_samples #módulo da FFT

#Compara os resultados
vect_np = optDFT_vect.ravel() - DFT_np.ravel() #empilha as matrizes e subtrai
print(f'VECT x numpy = {np.sum(vect_np)}')
partvect_np = DFT_vect.ravel() - DFT_np.ravel() #empilha as matrizes e subtrai
print(f'PART VECT x numpy = {np.sum(partvect_np)}')
iter_np = DFT_it.ravel() - DFT_np.ravel() #empilha as matrizes e subtrai
print(f'ITER x numpy = {np.sum(iter_np)}')