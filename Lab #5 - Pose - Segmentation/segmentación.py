import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolo11n-seg.pt')

def monitor_avanzado():
    cap = cv2.VideoCapture(0)

    historial_libre = []
    ventana_suavizado = 10 

    kernel = np.ones((5,5), np.uint8)

    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        if not success: break

        h, w = frame.shape[:2]
        results = model(frame, stream=True, conf=0.45, verbose=False)

        mask_ocupada = np.zeros((h, w), dtype=np.uint8)

        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for mask in masks:
                    m_resized = cv2.resize(mask, (w, h))
                    m_binary = (m_resized > 0.5).astype(np.uint8) * 255
                    
                    m_binary = cv2.morphologyEx(m_binary, cv2.MORPH_CLOSE, kernel)
                    
                    mask_ocupada = cv2.bitwise_or(mask_ocupada, m_binary)

        pixels_totales = h * w
        pixels_ocupados = np.count_nonzero(mask_ocupada)
        libre_actual = 100 - (pixels_ocupados / pixels_totales * 100)

        historial_libre.append(libre_actual)
        if len(historial_libre) > ventana_suavizado:
            historial_libre.pop(0)
        promedio_libre = sum(historial_libre) / len(historial_libre)

        overlay = frame.copy()
        overlay[mask_ocupada == 0] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        color_barra = (0, 255, 0) if promedio_libre > 40 else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (300, 60), (50, 50, 50), -1) 
        cv2.putText(frame, f"ESPACIO LIBRE: {promedio_libre:.1f}%", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_barra, 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (w-100, 30), 1, 1, (255, 255, 255), 1)

        cv2.imshow('Sistema de Gestion de Aforo Optimizada', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_avanzado()
