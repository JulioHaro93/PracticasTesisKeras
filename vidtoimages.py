import cv2
import os

# Ruta del video


frame_count = 0
for i in range(0,3):

    video_path = "C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptophlebiidae/homogeneas/video{}.mp4".format(i)
    
    # Carpeta de destino donde se guardarán los frames
    place = "C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptophlebiidae/homogeneas"
    
    # Asegúrate de que la carpeta exista (opcional, ya existe en este caso)
    os.makedirs(place, exist_ok=True)
    
    # Abre el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error al abrir el video.")
    else:
        print("✅ Video abierto correctamente.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # fin del video
              
            # Guarda cada frame en la carpeta 'place'
            frame_path = os.path.join(place, f'frame_varios_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
    
        cap.release()
        print(f"✅ {frame_count} cuadros guardados en '{place}'")
    
    i+=1
    print("//////////////////////////RUTA///////////////////////////")
    #print("/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/Frasco 229/Sam-cel")
    print("//////////////////////////RUTA///////////////////////////\n")
    print("####################################################")
    print("VIDEO{}".format(i))
    print("#####################################################")
    
"""
frame_count = 0
for i in range(0,6):

    video_path = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Oscar-cel/vid{}.mp4".format(i)
    
    # Carpeta de destino donde se guardarán los frames
    place = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Oscar-cel"
    
    # Asegúrate de que la carpeta exista (opcional, ya existe en este caso)
    os.makedirs(place, exist_ok=True)
    
    # Abre el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error al abrir el video.")
    else:
        print("✅ Video abierto correctamente.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # fin del video
            
            # Guarda cada frame en la carpeta 'place'
            frame_path = os.path.join(place, f'frame_Oscar_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
    
        cap.release()
        print(f"✅ {frame_count} cuadros guardados en '{place}'")
    
    i+=1
    print("//////////////////////////RUTA///////////////////////////")
    print("/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Oscar-cel")
    print("//////////////////////////RUTA///////////////////////////\n")
    print("####################################################")
    print("VIDEO{}".format(i))
    print("#####################################################")
    
    
frame_count = 0
for i in range(0,6):

    video_path = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Sam-cel/vid{}.mp4".format(i)
    
    # Carpeta de destino donde se guardarán los frames
    place = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Sam-cel"
    
    # Asegúrate de que la carpeta exista (opcional, ya existe en este caso)
    os.makedirs(place, exist_ok=True)
    
    # Abre el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error al abrir el video.")
    else:
        print("✅ Video abierto correctamente.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # fin del video
            
            # Guarda cada frame en la carpeta 'place'
            frame_path = os.path.join(place, f'frame_Sam_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
    
        cap.release()
        print(f"✅ {frame_count} cuadros guardados en '{place}'")
    
    i+=1
    print("//////////////////////////RUTA///////////////////////////")
    print("/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/Frasco 11/Sam-cel")
    print("//////////////////////////RUTA///////////////////////////\n")
    print("####################################################")
    print("VIDEO{}".format(i))
    print("#####################################################")
    """
    