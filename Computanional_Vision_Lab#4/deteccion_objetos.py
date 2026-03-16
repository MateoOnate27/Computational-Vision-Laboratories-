import cv2
import math
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

capture = cv2.VideoCapture(0)

classNames = ["persona", "bicicleta", "carro", "moto", "avion", "bus", "tren", "camion", "bote",
              "semaforo", "hidrante", "stop", "parquimetro", "banca", "pajaro", "gato",
              "perro", "caballo", "oveja", "vaca", "elefante", "oso", "zebra", "jirafa", "mochila", "paraguas",
              "cartera", "corbata", "maleta", "frisbee", "skis", "snowboard", "balon", "cometa", "bate",
              "guante", "skate", "surf", "raqueta", "botella", "copa", "taza", "tenedor", "cuchillo", "cuchara",
              "tazon", "banana", "manzana", "sandwich", "naranja", "broccoli", "zanahoria", "hot dog", "pizza",
              "donut", "pastel", "silla", "sofa", "planta", "cama", "comedor", "baño", "tv", "laptop",
              "mouse", "control", "teclado", "celular", "microondas", "horno", "tostador", "lavabo", "refrigerador",
              "libro", "reloj", "florero", "tijeras", "peluche", "secador", "cepillo"]

while True:
    success, img = capture.read()
    if not success:
        break

    # 3. Inferencia optimizada para CPU
    # imgsz=320 reduce la resolución para ganar mucha velocidad en tu Intel HD
    results = model(img, stream=True, imgsz=320, conf=0.4)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}'

            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Deteccion YOLO - Grafica Intel', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()