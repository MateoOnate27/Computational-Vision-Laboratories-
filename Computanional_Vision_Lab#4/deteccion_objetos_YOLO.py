import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

clases_seleccionadas = [63, 67, 73]

nombres_es = {63: "Laptop", 67: "Celular", 73: "Libro"}

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    if not success:
        break

    results = model(img, stream=True, imgsz=320, conf=0.4, classes=clases_seleccionadas)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f'{nombres_es[cls]} {conf:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Aplicacion de Deteccion Especifica', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()