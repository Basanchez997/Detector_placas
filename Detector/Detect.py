# importar librerias
import torch
import cv2
import numpy as np
from PIL import Image
import pytesseract

torch.cuda.is_available()
contatof = 0
# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:/Detector/model/placas.pt')

# Video captura
video = 0
# video = "http://192.168.0.128:4747/video"
#video = "http://192.168.1.31:9797/videostream.cgi?user=admin&pwd=am3ricas"
#video = "C:/Detector/vid1.mp4"
#video = "http://192.168.0.128:4747/video"

cap = cv2.VideoCapture(video)

x1 = 10
y1 = 10
xf = 10
yf = 10
id = 0
# Empezamos funciones
#Asignar brillo a la iamgen segun sus datos
def convertirEscala(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    #print(alpha, beta)
    return new_img.astype(np.uint8)


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[255],[0,255])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    #print("alpha antes " , alpha)
    #print("Beta antes", beta)
    if beta <-70 :
        alpha = alpha
        beta = beta/2
    elif beta<-7 :
        alpha = alpha
        beta = beta-50
    else:
        alpha = alpha
        beta = beta
    auto_result = convertirEscala(image, alpha=alpha, beta=beta)
    #print("alpha", alpha)
    #print("Beta", beta)
    return (auto_result)



#Procesamiento video
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read()

    # 10 fotoframas por segundo
    contatof += 1
    if contatof % 3 != 0:
        continue

    # Realizamos las detecciones
    detect = model(frame)

    # Extraemos la info
    info = detect.pandas().xyxy[0].to_dict(orient="records")  # predictions

    #info = detect_model(frame, id)
    #id += 1
    placa = frame.copy()
    if len(info) != 0:

        # Creamos un for
        for result in info:
            conf = result['confidence']
            if conf >= 0.70:
                # Clase
                cls = int(result['class'])
                # x1
                x1 = int(result['xmin'])
                # y1
                y1 = int(result['ymin'])
                # xf
                xf = int(result['xmax'])
                # yf
                yf = int(result['ymax'])

                cv2.rectangle(frame, (x1, y1), (xf, yf), (0, 255, 0), 2)

                y2 = y1 + 19  #Recorte arriba
                yf2 = yf - 29 #recorta de abajo
                x2 = x1 + 9  #recorte a la izq
                xf2 = xf + 5
                #recprte a la derec

                placa = placa[y2:yf2, x2:xf2]
                #cv2.imshow('placa', placa)

                # Extraemos el anocho  y el alto
                alp, anp, cp = placa.shape

                #cv2.imshow('Detector de Mva', Mva)
                placa_bin = placa.copy()
                placa_bin = automatic_brightness_and_contrast(placa_bin)
                #Creamos un frame pero de placa en escala de Grises
                gray = cv2.cvtColor(placa_bin, cv2.COLOR_BGR2GRAY)
                #Binarisamos para darle mas fuerza al color negro
                _, bin = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY)
                cv2.imshow('placa_prueba21', placa_bin)
                cv2.imshow('placa_prueba', bin)

                bin = bin.reshape(alp, anp)
                bin = Image.fromarray(bin)
                bin = bin.convert("L")
                bin.save("filename.jpeg")

                if alp >= 30 and anp >= 82:
                    # Declaramos la direccion de Pyresseract
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    # cv2.imshow('Detector de Mva', Mva)
                    # Extraemos el texto
                    config = "--psm 1 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    #config = "--psm 1"
                    texto = pytesseract.image_to_string(bin, config=config)
                    #print("aui", texto)

                    # If para no mostrar basura
                    if len(texto) >= 6:
                        Ctexto = texto
                        print(Ctexto)
                break


    # Mostramos FPS
    cv2.imshow('Detector de Carros', frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()