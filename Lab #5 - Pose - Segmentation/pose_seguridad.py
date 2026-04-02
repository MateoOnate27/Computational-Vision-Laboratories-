import cv2
from ultralytics import YOLO
import time

model = YOLO('yolo11n-pose.pt') 

cap = cv2.VideoCapture(0)

inicio_alerta = None
tiempo_requerido = 6.0 
alerta_confirmada = False

print("Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True, verbose=False, conf=0.5)
    
    pose_detectada_ahora = False

    for r in results:
        frame = r.plot() 
        
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            points = r.keypoints.xy[0].cpu().numpy()
            
            if len(points) > 10:
                ojo_izq_y, ojo_der_y = points[1][1], points[2][1]
                muñeca_izq_y, muñeca_der_y = points[9][1], points[10][1]

                if (0 < muñeca_izq_y < ojo_izq_y) and (0 < muñeca_der_y < ojo_der_y):
                    pose_detectada_ahora = True

    if pose_detectada_ahora:
        if inicio_alerta is None:
            inicio_alerta = time.time()
        
        duracion = time.time() - inicio_alerta
        if duracion >= tiempo_requerido:
            alerta_confirmada = True
            color_tema = (0, 0, 255) 
            msg = "!!! AMENAZA CONFIRMADA !!!"
        else:
            alerta_confirmada = False
            color_tema = (0, 165, 255) 
            msg = f"ANALIZANDO INTENCION: {int((duracion/tiempo_requerido)*100)}%"
    else:
        inicio_alerta = None
        alerta_confirmada = False
        color_tema = (0, 255, 0) 
        msg = "VIGILANCIA ACTIVA - NORMAL"

    cv2.rectangle(frame, (0, 0), (640, 45), (20, 20, 20), -1)
    cv2.putText(frame, msg, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_tema, 2)
    
    if inicio_alerta and not alerta_confirmada:
        bar_w = int((duracion / tiempo_requerido) * 640)
        cv2.rectangle(frame, (0, 45), (bar_w, 52), color_tema, -1)

    cv2.imshow("Seguridad Inteligente", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
