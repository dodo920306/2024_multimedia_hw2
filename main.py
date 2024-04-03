from librosa import load, display, frames_to_time
from librosa.display import waveshow
from librosa.feature import zero_crossing_rate
from sys import argv
from matplotlib.pyplot import show, savefig, subplots, tight_layout
from numpy import array
from math import ceil


fig, ax = subplots(3, 1)
x, sr = load(argv[1], sr=None)
print(x.max())
ax[0].plot(frames_to_time(range(len(x)), sr = sr, hop_length = 1), x)

energy = array([
    sum(x[i: i + 1024] ** 2)
    for i in range(0, len(x), 512)
])
t = frames_to_time(range(len(energy)), sr = sr)
ax[1].plot(t, energy)

# add a small constant to suppress the zcr near the beginning
ax[2].plot(t, zero_crossing_rate(x + 0.0001)[0])

ax[0].set_title('Waveform')
ax[1].set_title('Energy contour')
ax[2].set_title('Zero-crossing rate contour')
tight_layout()
savefig("results.png")
show()
