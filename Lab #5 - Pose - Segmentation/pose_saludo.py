import cv2
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt') 

cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)

    for r in results:
        annotated_frame = r.plot()
        
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            points = r.keypoints.xy[0].cpu().numpy()
            
            if len(points) > 10:
                oreja_izq_y = points[3][1]
                oreja_der_y = points[4][1]
                muñeca_izq_y = points[9][1]
                muñeca_der_y = points[10][1]

                saludo_izq = muñeca_izq_y < oreja_izq_y and muñeca_izq_y != 0
                saludo_der = muñeca_der_y < oreja_der_y and muñeca_der_y != 0

                if saludo_izq or saludo_der:
                    texto = "SALUDO DETECTADO"
                    color = (0, 255, 0) 
                else:
                    texto = "BUSCANDO SALUDO..."
                    color = (255, 255, 255) 

                cv2.rectangle(annotated_frame, (10, 10), (400, 60), (0,0,0), -1) 
                cv2.putText(annotated_frame, texto, (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Saludo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
