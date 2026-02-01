from ultralytics import YOLO

yolo = YOLO(model="../../../models/yolov8n.pt", task="detect")

result = yolo.predict(source="../../../data/Yolo/traffic.png")


print(type(result))
print(type(result[0]))
print(result[0].boxes)