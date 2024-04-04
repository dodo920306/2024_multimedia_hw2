from librosa import load, frames_to_time, pyin, note_to_hz, stft, amplitude_to_db
from librosa.display import specshow
from librosa.feature import zero_crossing_rate
from sys import argv
from matplotlib.pyplot import show, savefig, subplots, tight_layout, figure, colorbar, title, gcf
from numpy import array, mean, where, percentile, isnan, nan, median, abs, max
from scipy.ndimage import gaussian_filter1d


fig, ax = subplots(5, 1)
x, sr = load(argv[1], sr=None)
t = frames_to_time(range(len(x)), sr = sr, hop_length = 1)
ax[0].plot(t, x)
ax[3].plot(t, x)

energy = array([
    sum(x[i: i + 2048] ** 2)
    for i in range(0, len(x), 512)
])
t = frames_to_time(range(len(energy)), sr = sr)
ax[1].plot(t, energy)

# add a small constant to suppress the zcr near the beginning
zcr = zero_crossing_rate(x + 0.0001)[0]
ax[2].plot(t, zcr)

ITL = percentile(energy, 60)
ITU = percentile(energy, 80)
IZCT = mean(zcr)

ax[1].axhline(y = ITU, c = 'k')
ax[1].axhline(y = ITL, c = 'k')
ax[2].axhline(y = IZCT, c = 'k')

i = 0
while i < len(energy):
    while i < len(energy):
        if energy[i] > ITU:
            start_point = t[i]
            break
        else:
            i += 1
    j = i
    while j < len(energy):
        if energy[j] < ITL:
            end_point = t[j]
            break
        else:
            j += 1
    while i < j:
        if zcr[i] > 3 * IZCT:
            start_point = t[i]
            break
        else:
            i += 1
    ax[3].axvline(x = start_point, c = 'r')
    ax[3].axvline(x = end_point, c = 'g')
    i = j

f0, _, _ = pyin(x, fmin = note_to_hz('C2'), fmax = note_to_hz('C7'))
t = frames_to_time(range(len(f0)), sr = sr)
f0[isnan(f0)] = 0
f0 = gaussian_filter1d(f0, 3)
ax[4].plot(t, f0)

gcf().canvas.manager.set_window_title('Extract the speech features in Time Domain')

ax[0].set_title('Waveform')
ax[1].set_title('Energy contour')
ax[2].set_title('Zero-crossing rate contour')
ax[3].set_title('Endpoint Detection')
ax[4].set_title('Pitch Contour')
tight_layout()
savefig("result1.png")

figure()
D = stft(x)
S_db = amplitude_to_db(abs(D), ref = max)
specshow(S_db, x_axis = 'time', y_axis = 'log', sr = sr)
title("Spectrogram")
colorbar(format="%+2.f dB")
savefig("result2.png")
gcf().canvas.manager.set_window_title('Calculate the spectrogram in Frequency Domain')

show()
