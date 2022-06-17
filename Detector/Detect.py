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
# video = 0
video = "http://192.168.0.107:9797/videostream.cgi?user=admin&pwd=am3ricas"
#video = "http://192.168.1.31:9797/videostream.cgi?user=admin&pwd=am3ricas"
# video = "C:/Detector/vid1.mp4"
cap = cv2.VideoCapture(video)

x1 = 10
y1 = 10
xf = 10
yf = 10

# Empezamos
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read()

    # Correccion de color
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 10 fotoframas por segundo
    contatof += 1
    if contatof % 3 != 0:
        continue

    # Realizamos las detecciones
    detect = model(frame)

    # Extraemos la info
    info = detect.pandas().xyxy[0].to_dict(orient="records")  # predictions
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
                # print(x1,y1, xf,yf)

                y2 = y1 + 23  #Recorte arriba
                yf2 = yf - 30 #recorta de abajo
                x2 = x1 + 10  #recorte a la izq
                xf2 = xf - 10 #recprte a la derec



                placa = placa[y2:yf2, x2:xf2]

                frameHSV = cv2.cvtColor(placa, cv2.COLOR_BGR2HSV)
                mBph = np.matrix(frameHSV[:, :, 0])
                mGph = np.matrix(frameHSV[:, :, 1])
                mRph = np.matrix(frameHSV[:, :, 2])

                #print("mBph ", mBph, " mGph ", mGph, " mRph ", mRph),



                # Elegimos el umbral de verde en HSV
                umbral_bajo = (10, 150, 0)
                umbral_alto = (79, 155, 255)
                umbral_bajo2 = (0, 0, 188)
                umbral_alto2 = (0, 0, 0)
                # hacemos la mask y filtramos en la original
                mask = cv2.inRange(frameHSV, umbral_bajo, umbral_alto)
                mask2= cv2.inRange(frameHSV, umbral_bajo2, umbral_alto2)
                res = cv2.bitwise_and(frameHSV, frameHSV, mask=mask2)

                cv2.imshow('placa', mask)

                # Extraemos el anocho  y el alto
                alp, anp, cp = placa.shape

                # Procesar para eztraer los pixeles
                Mva = np.zeros((alp, anp))

                # Normalizamos las matrices

                mBp = np.matrix(placa[:, :, 0])
                mGp = np.matrix(placa[:, :, 1])
                mRp = np.matrix(placa[:, :, 2])

                Color2 = cv2.absdiff(mBp, mBp)
                # Binarizamos l aimagen
                #_, umbral2 = cv2.threshold(Color2, 60, 205, cv2.THRESH_BINARY)
                # umbral_limpio2 = cv2.dilate(umbral2, None, iterations=1)

                cv2.imshow('umbral_limpio2', Color2)

                # Creamos una mascara
                for col in range(0, alp):
                    for fil in range(0, anp):
                        Max = min(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                        Mva[col, fil] = 255 + Max
                # Binarizamos la imagen
                # _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)
                #_, bin = cv2.threshold(umbral2, 60, 205, cv2.THRESH_BINARY)

                # Convertimos la matriz en imagen



                # Validamos tener un buen tamaño de placas

                #cv2.imshow('Detector de Mva', Mva)
                placa_bin = placa.copy()


                gray = cv2.cvtColor(placa_bin, cv2.COLOR_BGR2GRAY)
                _, bin = cv2.threshold(gray, 60, 200, cv2.THRESH_BINARY)
                cv2.imshow('placa_prueba', gray)

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
                    print("aui", texto)

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
