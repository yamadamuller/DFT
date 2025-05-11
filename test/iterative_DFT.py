import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
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
t_init = time.time()
DFT_r_it, DFT_i_it = dft_utils.iterDFT(sig_i, sig_r)
print(f't_proc = {time.time()-t_init}s')
DFT_it = np.sqrt(DFT_r_it**2 + DFT_i_it**2)/n_samples #módulo

#plot
plt.figure(2, figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(t, sig_r)
plt.title('Domínio do tempo')
plt.xlabel('t(s)')
plt.ylabel('Sinal')

plt.subplot(2,1,2)
plt.plot(esc_freq, DFT_it)
plt.title('Domínio da frequência')
plt.xlabel('f(Hz)')
plt.ylabel('DFT')
plt.show()