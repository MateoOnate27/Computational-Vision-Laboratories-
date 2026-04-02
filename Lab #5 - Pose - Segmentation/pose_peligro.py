import cv2
import numpy as np
from ultralytics import YOLO

# 1. Cargamos el modelo de segmentación v11
model = YOLO('yolo11n-seg.pt') 

def monitoreo_botella_robusto():
    cap = cv2.VideoCapture(0)
    
    # Obtenemos dimensiones para el filtro de proximidad
    ret, frame_init = cap.read()
    if not ret: return
    H, W = frame_init.shape[:2]
    centro_pantalla = np.array([H/2, W/2])

    print("Sistema iniciado. Detectando Botella (Clase 39) y Persona (Clase 0)...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Inferencia con confianza alta (0.6) para limpiar el ruido del entorno
        results = model(frame, stream=True, conf=0.6)

        best_person_mask = None
        min_dist_person = float('inf')
        
        best_bottle_mask = None
        min_dist_bottle = float('inf')

        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                boxes_xywh = r.boxes.xywh.cpu().numpy()

                for i, cls in enumerate(classes):
                    # Lógica de cercanía al centro para ignorar a los demás
                    centro_obj = np.array([boxes_xywh[i][1], boxes_xywh[i][0]])
                    distancia = np.linalg.norm(centro_pantalla - centro_obj)

                    # Redimensionar máscara al tamaño de la cámara
                    m = cv2.resize(masks[i], (W, H))
                    m = (m > 0.5).astype(np.uint8)

                    if cls == 0 and distancia < min_dist_person: # Persona
                        min_dist_person = distancia
                        best_person_mask = m
                    
                    if cls == 39 and distancia < min_dist_bottle: # Botella
                        min_dist_bottle = distancia
                        best_bottle_mask = m

        # --- RECONOCIMIENTO DE LA ACTIVIDAD ---
        if best_person_mask is not None and best_bottle_mask is not None:
            # Intersección píxel a píxel (Solo posible con segmentación)
            contacto = cv2.bitwise_and(best_person_mask, best_bottle_mask)
            
            # Dibujar contornos para la demo
            cnts_p, _ = cv2.findContours(best_person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts_p, -1, (255, 0, 0), 2) # Azul: Persona
            
            cnts_b, _ = cv2.findContours(best_bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts_b, -1, (0, 165, 255), 2) # Naranja: Botella

            if np.sum(contacto) > 20: # Si hay píxeles compartidos
                # Pintar la zona de agarre de color verde brillante
                frame[contacto > 0] = [0, 255, 0]
                cv2.putText(frame, "ACTIVIDAD: BOTELLA SUJETADA", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "SISTEMA: EN ESPERA", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Buscando Persona y Botella...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow('Demo Segmentacion Botella - Mateo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitoreo_botella_robusto()
