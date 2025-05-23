import cv2
import sys

import matplotlib as plt

# Abre el video
try:
    #i=0
    for i in range(13):
        
        cap = cv2.VideoCapture('/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Beateidae/Julio-Cel/Frasco  472/vid1.mp4'.format(i))
        
        # Verifica si el video fue abierto correctamente
        if not cap.isOpened():
            print("Error al abrir el video.")
            sys.exit()
        
        frame_count = 0
        while True:
            # Lee un cuadro (frame) del video
            ret, frame = cap.read()
            
            if not ret:
                break  # Si no hay más cuadros, se detiene
            
            # Guarda el cuadro como una imagen
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)
            
            frame_count += 1
        
        # Libera los recursos
        cap.release()
        cv2.destroyAllWindows()
        #i+=1

except Exception as err:
    print(f"Ocurrió un error: {err}")
