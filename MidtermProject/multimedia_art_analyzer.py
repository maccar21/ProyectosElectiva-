import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import librosa
import librosa.display

# ------------------ Configuración ------------------
IMAGE_PATH = "espacio.jpg"   # Ruta de la imagen de entrada
AUDIO_PATH = "rock.wav"      # Ruta del audio de entrada
N_COLORES = 4                # Número de colores dominantes a extraer
OUTPUT_DIR = "outputs"       # Carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Crea la carpeta si no existe

# ------------------ PROCESAMIENTO DE IMAGEN ------------------
print("\n===  CARACTERÍSTICAS DE LA IMAGEN ===")

# Cargar imagen y convertir a RGB
img_bgr = cv2.imread(IMAGE_PATH)  # Lee la imagen en formato BGR (por defecto OpenCV)
if img_bgr is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {IMAGE_PATH}")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convertimos a RGB para análisis
h, w, _ = img.shape  # Obtenemos dimensiones
pixels = img.reshape(-1, 3).astype(np.float32)  # Aplanamos todos los píxeles

# Calcular histogramas RGB
hist_r, _ = np.histogram(pixels[:, 0], bins=256, range=(0, 255))  # Canal rojo
hist_g, _ = np.histogram(pixels[:, 1], bins=256, range=(0, 255))  # Canal verde
hist_b, _ = np.histogram(pixels[:, 2], bins=256, range=(0, 255))  # Canal azul

# Calcular histograma del tono (Hue)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convertimos a HSV
hue = img_hsv[:, :, 0].flatten()                # Extraemos solo el tono (Hue)
hist_hue, _ = np.histogram(hue, bins=36, range=(0, 180))

# Colores dominantes con KMeans
sample_size = 10000  # Número máximo de píxeles para muestreo
if pixels.shape[0] > sample_size:
    idx = np.random.choice(pixels.shape[0], sample_size, replace=False)  # Selecciona muestra aleatoria
    sample = pixels[idx]
else:
    sample = pixels
kmeans = KMeans(n_clusters=N_COLORES, random_state=42, n_init=10).fit(sample)  # Ejecuta clustering
centros = kmeans.cluster_centers_.astype(int)  # Colores promedio de cada clúster
etiquetas = kmeans.predict(pixels)            # Etiquetar todos los píxeles
conteos = np.bincount(etiquetas, minlength=N_COLORES)  # Cuenta píxeles por color
freqs = conteos / conteos.sum()               # Normaliza frecuencias
orden = np.argsort(-freqs)                    # Ordenar por frecuencia
centros = centros[orden]                      # Reordenar colores
freqs = freqs[orden]
centros_hsv = cv2.cvtColor(centros.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)

# Mostrar resultados en consola
print(f"- Tamaño de la imagen: {w}x{h} píxeles")
for i, (c, h, f) in enumerate(zip(centros, centros_hsv, freqs)):
    print(f"  Color {i+1}: RGB={c.tolist()} | HSV={tuple(int(x) for x in h)} | Frecuencia={f:.2f}")

# ------------------ PROCESAMIENTO DE AUDIO ------------------
print("\n===  CARACTERÍSTICAS DEL AUDIO ===")

# Cargar audio y obtener propiedades
y, sr = librosa.load(AUDIO_PATH, sr=None)  # Cargar con frecuencia original
duracion = librosa.get_duration(y=y, sr=sr)  # Duración total
n_channels = 1 if y.ndim == 1 else y.shape[1]  # Detectar mono o estéreo
print(f"- Frecuencia de muestreo: {sr} Hz")
print(f"- Duración: {duracion:.2f}s")
print(f"- Canales: {n_channels}")

# Eliminar silencios del audio
intervalos = librosa.effects.split(y, top_db=20)
if len(intervalos) > 0:
    audio_filtrado = np.concatenate([y[s:e] for s, e in intervalos])  # Une fragmentos no silenciosos
else:
    audio_filtrado = y.copy()

# Parámetros de análisis de audio
hop_length = 512
frame_length = 1024

# Energía RMS
rms = librosa.feature.rms(y=audio_filtrado, frame_length=frame_length, hop_length=hop_length)[0]
mean_rms = float(np.mean(rms))
print(f"- Energía RMS promedio: {mean_rms:.4f}")

# Centroide espectral
centroide = librosa.feature.spectral_centroid(y=audio_filtrado, sr=sr, n_fft=2048, hop_length=hop_length)[0]
mean_centroid = float(np.mean(centroide))
print(f"- Centroide espectral promedio: {mean_centroid:.1f} Hz")

# Tempo y beats
tempo, beats = librosa.beat.beat_track(y=audio_filtrado, sr=sr, hop_length=hop_length)
tempo = float(np.atleast_1d(tempo)[0])  # Forzamos escalar
print(f"- Tempo estimado: {tempo:.1f} BPM")

# Tasa de onsets
onset_env = librosa.onset.onset_strength(y=audio_filtrado, sr=sr, hop_length=hop_length)
onset_rate = float(np.mean(onset_env))
print(f"- Tasa de onsets promedio: {onset_rate:.2f}")

# Mel-espectrograma
S = librosa.feature.melspectrogram(y=audio_filtrado, sr=sr, n_mels=128, hop_length=hop_length)
S_db = librosa.power_to_db(S, ref=np.max)

# ------------------ NORMALIZACIÓN Y FUSIÓN ------------------
print("\n===  FUSIÓN MULTIMODAL ===")

# Calcular métricas normalizadas
mean_hue = float(np.mean(centros_hsv[:, 0]))
mean_sat = float(np.mean(centros_hsv[:, 1]))
calidez = float((180.0 - mean_hue) / 180.0)  # Escala tono a calidez
saturacion = float(mean_sat / 255.0)         # Escala saturación
nyquist = sr / 2.0
brillo = float(np.clip(mean_centroid / nyquist, 0.0, 1.0))  # Brillo en función del centroide
energia = float(np.clip(np.mean(rms) / (np.max(rms) + 1e-12), 0.0, 1.0))

fusion = { 'calidez': calidez, 'saturacion': saturacion, 'brillo': brillo, 'energia': energia }
print(f"- Calidez: {calidez:.2f}")
print(f"- Saturación: {saturacion:.2f}")
print(f"- Brillo: {brillo:.2f}")
print(f"- Energía: {energia:.2f}")

# ------------------ VISUALIZACIONES ------------------
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
import matplotlib.gridspec as gridspec
spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

# Imagen original
ax_img = fig.add_subplot(spec[0:2, 0])
ax_img.imshow(img)
ax_img.axis('off')
ax_img.set_title('Imagen Original')

# Colores dominantes como paleta
ax_sw = fig.add_subplot(spec[0, 1])
swatch = np.zeros((50, 300, 3), dtype=np.uint8)
start = 0
for c, f in zip(centros, freqs):
    w_rect = int(300 * f)   # ancho proporcional a frecuencia
    swatch[:, start:start + w_rect, :] = c
    start += w_rect
ax_sw.imshow(swatch)
ax_sw.axis('off')
ax_sw.set_title('Colores Dominantes (RGB)')

# Tabla resumen de colores
ax_tab = fig.add_subplot(spec[1, 1])
ax_tab.axis('off')
filas = []
for i, (c, h, fr) in enumerate(zip(centros, centros_hsv, freqs)):
    filas.append([f"C{i+1}", str(c.tolist()), f"HSV:{tuple(int(x) for x in h)}", f"{fr:.2f}"])
columnas = ['Id', 'RGB', 'HSV', 'Frecuencia']
tabla = ax_tab.table(cellText=filas, colLabels=columnas, loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
ax_tab.set_title('Resumen de Colores')

# Histogramas RGB
ax_hist = fig.add_subplot(spec[2, 0])
bins = np.arange(256)
ax_hist.plot(bins, hist_r, label='Rojo', alpha=0.7)
ax_hist.plot(bins, hist_g, label='Verde', alpha=0.7)
ax_hist.plot(bins, hist_b, label='Azul', alpha=0.7)
ax_hist.set_xlim(0, 255)
ax_hist.set_title('Histogramas RGB')
ax_hist.set_xlabel('Intensidad')
ax_hist.set_ylabel('Frecuencia')
ax_hist.legend()

# Histograma de tono (Hue)
ax_hue = fig.add_subplot(spec[2, 1])
ax_hue.bar(np.linspace(0, 180, len(hist_hue)), hist_hue, width=5)
ax_hue.set_title('Histograma de Tono (HSV)')
ax_hue.set_xlabel('Tono (Hue)')
ax_hue.set_ylabel('Frecuencia')

# Forma de onda del audio
ax_wf = fig.add_subplot(spec[0, 2])
librosa.display.waveshow(audio_filtrado, sr=sr, ax=ax_wf)
ax_wf.set_title('Forma de onda (sin silencios)')
ax_wf.set_xlabel('Tiempo (s)')
ax_wf.set_ylabel('Amplitud')

# RMS por frame
ax_rms = fig.add_subplot(spec[1, 2])
ax_rms.plot(librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length), rms)
ax_rms.set_title('Energía RMS (por frames)')
ax_rms.set_xlabel('Tiempo (s)')
ax_rms.set_ylabel('Intensidad')

# Centroide espectral
ax_cent = fig.add_subplot(spec[2, 2])
ax_cent.plot(librosa.frames_to_time(np.arange(len(centroide)), sr=sr, hop_length=hop_length), centroide)
ax_cent.set_title(f'Centroide Espectral (promedio {mean_centroid:.1f} Hz)')
ax_cent.set_xlabel('Tiempo (s)')
ax_cent.set_ylabel('Frecuencia (Hz)')

# Guardar figura resumen
fig.suptitle('Perfil Artístico Multimedia', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUTPUT_DIR, 'perfil_multimedia.png'), dpi=200)

# Mel-espectrograma
plt_fig2 = plt.figure(figsize=(12, 4))
ax_mel = plt_fig2.add_subplot(1, 1, 1)
librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, ax=ax_mel)
ax_mel.set_title('Mel-Espectrograma (dB)')
ax_mel.set_xlabel('Tiempo (s)')
ax_mel.set_ylabel('Frecuencia Mel')
plt_fig2.tight_layout()
plt_fig2.savefig(os.path.join(OUTPUT_DIR, 'mel_espectrograma.png'), dpi=200)

# ------------------ INTERPRETACIÓN ------------------
interpretacion = (
    f"Interpretación (máx.200 palabras):\n\n"
    f"La paleta dominante tiene HSV promedio (tono≈{mean_hue:.1f}, sat≈{mean_sat:.0f}), " 
    f"lo que sugiere tonos {'cálidos' if calidez > 0.55 else 'fríos/modulados'} y " 
    f"{'alta' if saturacion > 0.6 else 'baja/moderada'} saturación.\n"
    f"El audio muestra un centroide medio de {mean_centroid:.1f} Hz y energía RMS "
    f"compatible con nivel {'alto' if energia > 0.5 else 'contenido'}.\n\n"
    f"Fusión normalizada (calidez={calidez:.2f}, saturación={saturacion:.2f}, "
    f"brillo={brillo:.2f}, energía={energia:.2f}) sugiere "
    f"una correspondencia entre color y timbre: colores más cálidos/saturados "
    f"se asocian a un timbre más brillante y energético en esta pieza."
)
# Guardamos la interpretación en archivo de texto
with open(os.path.join(OUTPUT_DIR, 'interpretacion.txt'), 'w', encoding='utf-8') as f:
    f.write(interpretacion)

print("\nProcesamiento completado. Archivos guardados en 'outputs/'.")
