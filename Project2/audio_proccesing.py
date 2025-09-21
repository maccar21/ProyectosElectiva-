import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

#Recording a Voice Sample
#Se carga el archivo de audio y obtiene la tasa de muestreo y los datos de audio
audio_file = "audio.wav"
y, sr = librosa.load(audio_file, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
n_channels = 1 if y.ndim == 1 else y.shape[1]

#Filtering the Silent Intervals
non_silent_intervals = librosa.effects.split(y, top_db=20)#elimina los silencios del audio
print(f"Sample Rate: {sr}KHz, Duration: {duration}s, Channels: {n_channels}") #imprime la tasa de muestreo, duracion y numero de canales
print(f"Non-silent intervals: {non_silent_intervals}") #imprime los intervalos no silenciosos

if len(non_silent_intervals) > 0:
    filtered_audio = np.concatenate([y[start:end] for start, end in non_silent_intervals])
else :
    filtered_audio = y #si no hay silencio, se usa el audio original

y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000) #resamplea el audio de 44100kHz a 16kHz

#Sampling, Quantization, and Coding
desired_sample_rate = 8000 #se define una tasa de muestreo deseada
bit_depth = 8  #se define la profundidad de bits

resampled_audio = librosa.resample(filtered_audio, orig_sr=sr, target_sr=desired_sample_rate) #resamplea el audio al sample rate deseado

normalized_audio = resampled_audio / np.max(np.abs(resampled_audio)) #normaliza el audio entre -1 y 1

max_amplitude = 2**(bit_depth - 1) - 1 
quantized_audio = np.round(normalized_audio * max_amplitude).astype(np.int16) #cuantiza el audio a la profundidad de bits deseada


#Fourier Transformation
N = len(filtered_audio)   #numero de muestras
T = 1.0 / sr              #intervalo de muestreo
yf = fft(filtered_audio)  #se ejecuta la transformada de fourier rapidamente
xf = fftfreq(N, T)[:N//2] #frecuencias positivas

amplitude = 2.0/N * np.abs(yf[0:N//2]) #calcula el espectro de amplitud

eps = 1e-10 #valor pequeño para evitar log(0)

power_watts = (amplitude / np.sqrt(2))**2 / 1.0 #calcula la potencia en vatios
power_mw = power_watts * 1000  #convierte la potencia a miliwatts
amplitude_dbm = 10 * np.log10(power_mw + eps)  #convierte la potencia a dBm

mask = (xf >= 0) & (xf <= 400) #filtro para frecuencias entre 0 y 400 Hz
xf_plot = xf[mask] 
amplitude_dbm_plot = amplitude_dbm[mask] #aplica el filtro

# Visualizing the Frequency Histogram
plt.figure(figsize=(10,6))
plt.hist(xf_plot, bins=50, weights=amplitude_dbm_plot, color='skyblue', edgecolor='black')
plt.title("Histograma de Frecuencias del Audio")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Intensidad Acumulada (dBm)")
plt.show()

dominant_freq = xf_plot[np.argmax(amplitude_dbm_plot)] #frecuencia con la mayor amplitud
print(f"Frecuencia normal/dominante: {dominant_freq:.2f} Hz")

cumulative_power = np.cumsum(amplitude_dbm_plot - np.min(amplitude_dbm_plot)) # energía acumulada
cumulative_power /= cumulative_power[-1]  # normalizar a [0,1]

low_idx = np.argmax(cumulative_power >= 0.05)   # 5% energía
high_idx = np.argmax(cumulative_power >= 0.95)  # 95% energía

freq_range = (xf_plot[low_idx], xf_plot[high_idx]) #rango de frecuencias significativas
print(f"Rango de frecuencias significativas: {freq_range[0]:.2f} Hz - {freq_range[1]:.2f} Hz")

plt.figure(figsize=(10, 6))
plt.plot(xf_plot, amplitude_dbm_plot, label= "Espectro de frecuencia (dBm)")

if 0 <= dominant_freq <= 400:
    plt.axvline(x=dominant_freq, color='r', linestyle='--',
                label=f"Frecuencia Normal/Dominante: {dominant_freq:.2f} Hz")

# Líneas verdes para el rango de frecuencias
plt.axvline(x=freq_range[0], color='g', linestyle='--', 
            label=f"Inicio rango: {freq_range[0]:.2f} Hz")
plt.axvline(x=freq_range[1], color='g', linestyle='--', 
            label=f"Fin rango: {freq_range[1]:.2f} Hz")

plt.title("Espectro de Frecuencia del Audio")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (dBm)")
plt.grid(True)
plt.legend()
plt.show()
















