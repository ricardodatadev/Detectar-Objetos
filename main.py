import cv2
import pytesseract
from ultralytics import YOLO

# Carrega o modelo YOLO
model = YOLO('yolo11n.pt')

# Abre o vídeo
video = cv2.VideoCapture('ex03.mp4')

# Define o caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

while video.isOpened():
    check, img = video.read()
    if not check:
        break  # Sai do loop se o vídeo terminar
    
    img = cv2.resize(img, (1280, 720))
    result = model(img)
    
    for r in result:
        boxes = r.boxes 
        clas = boxes.cls.int()
        xy = boxes.xyxy.int()
        
        for box, cls in zip(xy, clas):
            if cls == 7:
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 0), 5)
                
                # --- Captura da área dentro do retângulo ---
                corte = img[y1:y2, x1:x2]  # Recorta a imagem
                
                # --- Pré-processamento para melhorar OCR ---
                corte = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)  # Converte para tons de cinza
                corte = cv2.threshold(corte, 150, 255, cv2.THRESH_BINARY)[1]  # Binariza a imagem
                
                # --- Reconhecimento de texto ---
                texto = pytesseract.image_to_string(corte, config='--psm 6')
                
                # Exibe a área recortada e reconhecida
                cv2.imshow('Texto Cortado', corte)
                
                # Exibe o texto extraído
                print(f'Frota: {texto.strip()}')
    
    # Mostra a imagem principal com os retângulos
    cv2.imshow('Window Capture', img)
    
    # Espera por uma tecla (se apertar 'q', sai do loop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
video.release()
cv2.destroyAllWindows()

      
   
      
 








