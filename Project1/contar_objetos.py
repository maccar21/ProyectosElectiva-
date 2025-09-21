import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread("monedas.jpeg")
escalaGrises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)# se ajusta la escala de grises
suavizado = cv2.GaussianBlur(escalaGrises, (7, 7), 0) #Reducción de ruido con el suavizado gaussiano, se quira el ruido que rompe las imagenes


edges = cv2.Canny(suavizado, 50, 150) #Decidi usar Canny que Otsu ya que es mejor para objetos reales, detecta bordes de manera mas robusta antes las luces y sombras.
kernel = np.ones((9, 9), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) #Operaciones morfológicas usadas para cerrar huecos y unir contornos
                                                          #Uso de CLOSE ya que une líneas de bordes rotos y cierra huecos dentro de los objetos 

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Encontrar los contornos


valid_contours = [c for c in contours if cv2.contourArea(c) > 1000] #Filtrar el ruido o los contornos pequeños
                                                                    #Ya que son fotos tomadas por mi mismo en una mesa, se uso un area de 1000 px´2
                                                                    #para que de esta manera ignore manchas y solo cuente los objetos ms grabdes

output = imagen.copy()
for i, contour in enumerate(valid_contours): #Se cuentan todos los contornos validos
    x, y, w, h = cv2.boundingRect(contour)   #Se define un rectangulo que encierre el contorno 
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2) #Se dibuja el rectangulo
    cv2.putText(output, f"{i+1}", (x, y - 10),                        #Se numera cada objeto detectado 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #Dibujar resultados


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)), plt.title("Foto Original")
plt.subplot(1, 3, 2), plt.imshow(closed, cmap="gray"), plt.title("Foto Procesada")
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.title(f"Objetos detectados: {len(valid_contours)}")
plt.show() # Mostrar resultados
