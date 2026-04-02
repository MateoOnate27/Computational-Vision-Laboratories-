import cv2
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt') 

cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir")

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
                nariz_y = points[0][1]
                muñeca_izq_y = points[9][1]
                muñeca_der_y = points[10][1]

                if muñeca_izq_y < nariz_y and muñeca_der_y < nariz_y and muñeca_izq_y != 0:
                    cv2.putText(annotated_frame, "ALERTA: BRAZOS ARRIBA", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif muñeca_izq_y < nariz_y or muñeca_der_y < nariz_y:
                    cv2.putText(annotated_frame, "UNA MANO LEVANTADA", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("YOLOv11 Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
