# needed
import os
import re
import soundfile as sf
from IPython.display import Audio
from IPython.display import display
import numpy as np
# recommended ...
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
import scipy.io
import matplotlib.pyplot as plt

# read data
login = "XXXXXX"
zip_file = "known" + ".zip"
for name in ("known.zip", "valid.zip", login + ".zip"):
  file = "https://www.fit.vut.cz/study/course/ISS/public/proj2025-26/" + name

# load the known data - the function returns a big matrix with all the signals
def load_data (S, dirname, count, no_samples):
  ii = 0
  for one in np.arange(count):
    S[ii], Fs = sf.read(dirname + "/" + str(one) + ".wav")
    ii = ii+1

Fs = 16000
# load known data
N_known = 706; duration_known = 10; no_samples_known = Fs * duration_known
known_signals=np.zeros([N_known, no_samples_known]); load_data(known_signals, "known", N_known, no_samples_known)
#display(Audio(known_signals[45], rate=Fs))

# load validation data
N_valid = 50; duration_valid = 5; no_samples_valid = Fs * duration_valid
valid_signals=np.zeros([N_valid, no_samples_valid]); load_data(valid_signals, "valid", N_valid, no_samples_valid)
#display(Audio(valid_signals[45], rate=Fs))


def compute_similarity_matrix(query_signals, N_query, N_known, Fs):

  # výstupná matica podobností
  similarities = np.zeros((N_query, N_known), dtype=np.float32)

  # STFT parametre
  nperseg = 1024
  noverlap = 768
  hop = nperseg - noverlap

  # frekvenčné pásmo, ktoré porovnávame
  fmin = 300
  fmax = 3400

  # krok posuvu (v čase) pre sliding window po known spektrogramoch
  step_samples = int(Fs * 0.125)
  step_frames = max(1, int(round(step_samples / hop)))

  # pooling = zníženie rozlíšenia (spriemerovanie) -> rýchlejšie + robustnejšie
  pool = 2

  # predpočítanie valid: spravíme z nich jednotlivé vektory
  valid_unit = [None] * N_query
  valid_T = [0] * N_query

  for i in range(N_query):
    x = query_signals[i]

    #STFT
    f, t, S = signal.stft(x, fs=Fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    M = np.abs(S).astype(np.float32)

    M = np.log1p(M)

    # frekvenčný výrez
    mask = (f >= fmin) & (f <= fmax)
    M = M[mask, :]

    # normalizácia po frekvenčných kanáloch
    M = M / (np.mean(M, axis=1, keepdims=True) + 1e-10)
    M = M - np.mean(M, axis=1, keepdims=True)

    # pooling po frekvencii
    F = (M.shape[0] // pool) * pool
    M = M[:F, :]
    M = M.reshape(F // pool, pool, M.shape[1]).mean(axis=1)

    v = M.ravel()
    valid_unit[i] = v / (np.linalg.norm(v) + 1e-10)
    valid_T[i] = M.shape[1]

  # predspracovanie známych nahrávok
  known_specs = [None] * N_known
  known_T = [0] * N_known

  for j in range(N_known):
    y = known_signals[j]
    f2, t2, S2 = signal.stft(y, fs=Fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    K = np.abs(S2).astype(np.float32)

    K = np.log1p(K)

    mask2 = (f2 >= fmin) & (f2 <= fmax)
    K = K[mask2, :]

    K = K / (np.mean(K, axis=1, keepdims=True) + 1e-10)
    K = K - np.mean(K, axis=1, keepdims=True)

    F2 = (K.shape[0] // pool) * pool
    K = K[:F2, :]
    K = K.reshape(F2 // pool, pool, K.shape[1]).mean(axis=1)

    known_specs[j] = K
    known_T[j] = K.shape[1]

  # výpočet podobnosti pomocou posuvného okna
  for i in range(N_query):
    if i % 5 == 0:
      print(f"query {i}/{N_query}")

    v_unit = valid_unit[i]
    vT = valid_T[i]

    for j in range(N_known):
      K = known_specs[j]
      kT = known_T[j]

      if kT < vT:
        similarities[i, j] = -1e12
        continue

      best_sim = -1e12

      for start in range(0, kT - vT + 1, step_frames):
        seg = K[:, start:start+vT]
        seg_v = seg.ravel()
        seg_unit = seg_v / (np.linalg.norm(seg_v) + 1e-10)

        #skalárny súčin
        sim = np.dot(v_unit, seg_unit)

        if sim > best_sim:
          best_sim = sim

      similarities[i, j] = best_sim

  return similarities

# evaluation - the function produces Top-1 and Top-5 accuracy on validation data
def evaluate(scores, key):
  indices = np.flip(np.argsort(scores), axis=-1) # we want highest to lowest ...
  #print(scores[0,key[0]], key[0], indices)
  top1acc = np.sum(key == indices[:,0]) / indices.shape[0]
  top5acc = 0
  for ii in range(5):
    top5acc += np.sum(key == indices[:,ii])
  top5acc /=  indices.shape[0]
  return top1acc, top5acc

key = np.loadtxt("valid/key.txt", delimiter = ',', usecols=(1), dtype ='int')

# here comes the computation and evaluation on validation data ...
scores_valid = compute_similarity_matrix(valid_signals, N_valid, N_known, Fs)
#print (key)
top1, top5 = evaluate(scores_valid, key)
print("Top 1 accuracy ", top1 * 100, "%, Top 5 accuracy ", top5 * 100, "%" )


# load your data
N_eval = 50; duration_eval = 5; no_samples_eval = Fs * duration_eval
eval_signals=np.zeros([N_eval, no_samples_eval]); load_data(eval_signals, login, N_eval, no_samples_eval)
#display(Audio(valid_signals[45], rate=Fs))

scores_eval = compute_similarity_matrix(eval_signals, N_eval, N_known, Fs)

# export it
np.savetxt("eval.txt", scores_eval)

# Save evaluation similarity matrix

#Vizualizácia
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fs = 16000

vid = 15
best_known = np.argmax(scores_valid[vid])

x_known = known_signals[best_known]
x_valid = valid_signals[vid]

t_known = np.arange(len(x_known)) / Fs
t_valid = np.arange(len(x_valid)) / Fs

# Vizualizácia 2
import matplotlib.pyplot as plt

x = known_signals[0]
t = np.arange(len(x)) / Fs

plt.figure(figsize=(10, 3))
plt.plot(t, x)
plt.xlabel("Čas [s]")
plt.ylabel("Amplitúda")
plt.title("Časový priebeh známej hudobnej nahrávky")
plt.tight_layout()
plt.show()

# Vizualizácia 3
from scipy import signal

f, t, S = signal.stft(
    x,
    fs=Fs,
    nperseg=512,
    noverlap=256,
    boundary=None
)
S_log = 20 * np.log10(np.abs(S) + 1e-10)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, S_log, shading="gouraud")
plt.colorbar(label="Amplitúda [dB]")
plt.xlabel("Čas [s]")
plt.ylabel("Frekvencia [Hz]")
plt.title("Logaritmický spektrogram známej nahrávky")
plt.ylim(0, 6000)
plt.tight_layout()
plt.show()

# Vizualizácia 4
y = valid_signals[0]

f2, t2, S2 = signal.stft(
    y,
    fs=Fs,
    nperseg=512,
    noverlap=256,
    boundary=None
)
S2_log = 20 * np.log10(np.abs(S2) + 1e-10)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t2, f2, S2_log, shading="gouraud")
plt.colorbar(label="Amplitúda [dB]")
plt.xlabel("Čas [s]")
plt.ylabel("Frekvencia [Hz]")
plt.title("Logaritmický spektrogram testovacej nahrávky")
plt.ylim(0, 6000)
plt.tight_layout()
plt.show()

#Vizualizácia číslo 5
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(t_known, x_known)
plt.title(f"Známy signál (known[{best_known}])")
plt.xlabel("Čas [s]")
plt.ylabel("Amplitúda")

plt.subplot(1,2,2)
plt.plot(t_valid, x_valid)
plt.title(f"Validačný signál (valid[{vid}])")
plt.xlabel("Čas [s]")

plt.tight_layout()
plt.show()

#Vizualizácia číslo 6
plt.figure(figsize=(12,5))

for i, (x, title) in enumerate([
    (x_known, f"Spektrogram – known[{best_known}]"),
    (x_valid, f"Spektrogram – valid[{vid}]")
]):
    f, t, S = signal.stft(x, Fs, nperseg=512, noverlap=256)
    S_db = 20*np.log10(np.abs(S) + 1e-10)

    plt.subplot(1,2,i+1)
    plt.pcolormesh(t, f, S_db, shading='gouraud')
    plt.ylim(0, 5500)
    plt.title(title)
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvencia [Hz]")
    plt.colorbar(label="Amplitúda [dB]")

plt.tight_layout()
plt.show()