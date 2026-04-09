import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 1. Cargar el modelo más ligero (Nano)
model = YOLO('yolo26n.pt') 

video_path = "store.mp4"
cap = cv2.VideoCapture(video_path)

# Estructuras de datos
track_history = defaultdict(lambda: [])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

# --- VARIABLES DE OPTIMIZACIÓN ---
frame_count = 0
skip_frames = 3  # Procesa 1 de cada 3 cuadros para ganar velocidad
# ---------------------------------

# --- VARIABLES PARA EL CONTADOR ---
line_y = int(height * 0.60)  # Línea al 60% de la altura (cerca de la entrada)
counter = 0
already_counted = set()
# ----------------------------------

print("Procesando con optimización para CPU... Presiona 'q' para salir.")

# --- CONFIGURACIÓN DEL OUTPUT ---
output_path = "resultado_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para MP4
fps = cap.get(cv2.CAP_PROP_FPS) # Obtener FPS originales del video

# Definir el objeto de escritura
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# --------------------------------

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # OPTIMIZACIÓN 1: Saltar cuadros
    # Solo le pedimos a la IA que trabaje cada 'skip_frames'
    if frame_count % (skip_frames + 1) != 0:
        # Si saltamos el cuadro, solo mostramos el anterior rápido o lo ignoramos
        continue

    # OPTIMIZACIÓN 2: Redimensionar antes de procesar
    # Procesar una imagen más pequeña es MUCHO más rápido para la CPU
    frame_small = cv2.resize(frame, (640, 360)) 

    # Tracking (Solo personas, en CPU, y con imagen pequeña)
    results = model.track(frame_small, persist=True, classes=[0], device='cpu', verbose=False)

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        # Escalar las coordenadas de vuelta al tamaño original
        boxes = results[0].boxes.xywh.cpu().numpy()
        # Factor de escala (asumiendo original es 1280x720 o similar)
        scale_x = width / 640
        scale_y = height / 360
        
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            # Ajustar centro al tamaño real
            center = (int(x * scale_x), int(y * scale_y))

            # TASK 1: Motion Trail
            track = track_history[track_id]
            track.append(center)

            # LÓGICA DE CONTEO INTELIGENTE
            if len(track) >= 2:
                prev_y = track[-2][1]
                curr_y = center[1]

                # Detección de cruce (Entrada: de arriba hacia abajo)
                if prev_y < line_y <= curr_y and track_id not in already_counted:
                    counter += 1
                    already_counted.add(track_id)

            if len(track) > 15: # Estela más corta para no saturar
                track.pop(0)

            points = np.array(track).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

            # TASK 2: Heatmap
            cv2.circle(heatmap_accumulator, center, 20, 0.5, thickness=-1)

            # Dibujar la línea virtual
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

            # Mostrar el contador
            cv2.putText(annotated_frame, f"Entradas: {counter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Visualización del Heatmap
    heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    final_img = cv2.addWeighted(annotated_frame, 0.7, heatmap_color, 0.3, 0)

    out.write(final_img)

    # Mostrar resultado
    cv2.imshow("Tracking Fluido (CPU Optimized)", final_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
