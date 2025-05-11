import numpy as np

def optDFT(signal_r, signal_i, shift=True):
    '''
    :param signal_r: parte real do sinal
    :param signal_i: parte imaginária do sinal
    :param shift: flag para aplicar ou não shift na DFT
    :return parte real e imaginária do espectro do sinal
    '''
    N = len(signal_r)  #número total de amostras

    #garante que os sinais são no formato 1xN
    signal_r = np.atleast_2d(signal_r)
    signal_i = np.atleast_2d(signal_i)

    #Realiza o shift (se True)
    if shift:
        u = np.arange(0, N, 1) #vetor com componentes da frequência para o shift
        signal_r = signal_r*(-1)**u #shift na parte real
        signal_i = signal_i*(-1)**u #shift na parte imaginária

    # DFT vetorizada
    n = np.arange(0, N, 1) #intervalo 0:N-1
    kx, ky = np.meshgrid(n, n) #matriz 2D com o intervalo 0:N-1
    theta = 2*np.pi*ky*n/N #theta = (2*pi*u/N)*(n-1)
    dft_r = signal_r@np.cos(theta) + signal_i@np.sin(theta) #r[n]*cos(theta) + fi[n]*sen(theta)
    dft_i = signal_i@np.cos(theta) - signal_r@np.sin(theta) #j{-fr[n]*sen(theta) + fi[n]*cos(theta)}

    return dft_r.ravel(), dft_i.ravel() #retorna vetor 1D

def vectDFT(signal_r, signal_i, shift=True):
  '''
  :param signal_r: parte real do sinal
  :param signal_i: parte imaginária do sinal
  :param shift: flag para aplicar ou não shift na DFT
  :return parte real e imaginária do espectro do sinal
  '''
  N = len(signal_r) #número total de amostras
  dft_r = np.zeros((N,1)) #vetor para armazenar os valores reais do espectro
  dft_i = np.zeros_like(dft_r) #vetor para armazenar os valores imaginários do espectro

  #Realiza o shift (se True)
  if shift:
    u = np.arange(0,N,1) #vetor com componentes da frequência para o shift
    signal_r = signal_r*(-1)**u #shift na parte real
    signal_i = signal_i*(-1)**u #shift na parte imaginária

  #DFT parcialmente vetorizada
  n = np.arange(0,N,1) #vetor com os índices das amostras até N
  for k in n:
    theta = 2*np.pi*k*n/N #theta = (2*pi*u/N)*(n-1)
    sum_r = signal_r*np.cos(theta) + signal_i*np.sin(theta) #r[n]*cos(theta) + fi[n]*sen(theta)
    sum_i = signal_i*np.cos(theta) - signal_r*np.sin(theta) #j{-fr[n]*sen(theta) + fi[n]*cos(theta)}
    dft_r[k] = np.sum(sum_r) #componente real é a soma real
    dft_i[k] = np.sum(sum_i) #componente im. é a soma im.

  return dft_r, dft_i

def iterDFT(signal_r, signal_i, shift=True):
  '''
  :param signal_r: parte real do sinal
  :param signal_i: parte imaginária do sinal
  :param shift: flag para aplicar ou não shift na DFT
  :return parte real e imaginária do espectro do sinal
  '''
  N = len(signal_r) #número total de amostras
  dft_r = np.zeros((N,1)) #vetor para armazenar os valores reais do espectro
  dft_i = np.zeros_like(dft_r) #vetor para armazenar os valores imaginários do espectro

  #Realiza o shift (se True)
  if shift:
    u = np.arange(0,N,1) #vetor com componentes da frequência para o shift
    signal_r = signal_r*(-1)**u #shift na parte real
    signal_i = signal_i*(-1)**u #shift na parte imaginária

  #DFT iterativa
  n = np.arange(0,N,1) #vetor com os índices das amostras até N
  for k in n:
    curr_sum_r = 0 #variável para armazenar a soma iterativa da parte real
    curr_sum_i = 0 #variável para armazenar a soma iterativa da parte imaginária

    for i in n:
      theta = 2*np.pi*k*i/N #theta = (2*pi*u/N)*(n-1)
      curr_sum_r += signal_r[i]*np.cos(theta) + signal_i[i]*np.sin(theta) #r[n]*cos(theta) + fi[n]*sen(theta)
      curr_sum_i += signal_i[i]*np.cos(theta) - signal_r[i]*np.sin(theta) #j{-fr[n]*sen(theta) + fi[n]*cos(theta)}

    dft_r[k] = curr_sum_r #componente real é a soma real
    dft_i[k] = curr_sum_i #componente im. é a soma im.

  return dft_r, dft_i