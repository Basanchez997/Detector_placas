#importar librerias
import torch
import cv2
import numpy as np
from PIL import Image
import pytesseract

torch.cuda.is_available()
contatof = 0
#Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:/Detector/model/placas.pt')

#Video captura
#video = 0
#video = "http://192.168.0.128:4747/video"
#video = "http://192.168.1.31:9797/videostream.cgi?user=admin&pwd=am3ricas"
video = "C:/Detector/VIDEO.MOV"
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
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #10 fotoframas por segundo
    contatof += 1
    if contatof % 3 != 0:
        continue

    # Realizamos las detecciones
    detect = model(frame)


    #Extraemos la info
    info = detect.pandas().xyxy[0].to_dict(orient="records")  #  predictions

    if len(info) != 0:

        #Creamos un for
        for result in info:
            conf = result['confidence']
            if conf >= 0.70:
                #Clase
                cls = int(result['class'])
                #x1
                x1 = int(result['xmin'])
                # y1
                y1 = int(result['ymin'])
                # xf
                xf = int(result['xmax'])
                # yf
                yf = int(result['ymax'])

                cv2.rectangle(frame,(x1, y1), (xf, yf), (0, 255, 0), 2)
                #print(x1,y1, xf,yf)
                placa = frame[y1:yf, x1:xf]

                #Extraemos el anocho  y el alto
                alp, anp, cp = placa.shape



                #Procesar para eztraer los pixeles
                Mva = np.zeros((alp, anp))

                #Normalizamos las matrices

                mBp = np.matrix(placa[:, :, 0])
                mGp = np.matrix(placa[:, :, 1])
                mRp = np.matrix(placa[:, :, 2])

                #Creamos una mascara
                for col in range(0, alp):
                    for fil in range(0, anp):
                        Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                        Mva[col,fil] = 255 - Max
                #Binarizamos la imagen
                _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

                #Convertimos la matriz en imagen
                bin = bin.reshape(alp, anp)
                bin = Image.fromarray(bin)
                bin = bin.convert("1")
                cv2.imwrite('C:/Detector/filename.jpeg', Mva)


                #Validamos tener un buen tamaño de placas

                cv2.imshow('Detector de Mva', placa)
                if alp >= 36 and anp>=82:
                    # Declaramos la direccion de Pyresseract
                    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
                    cv2.imshow('Detector de Mva', Mva)
                    # Extraemos el texto
                    config = "--psm 1"
                    texto = pytesseract.image_to_string(bin, config=config)
                    print("aui", texto)

                    # If para no mostrar basura
                    if len(texto) >= 6:
                        Ctexto = texto
                        print(Ctexto)
                break



    # Mostramos FPS
    cv2.imshow('Detector de Carros', frame)
    #cv2.imshow('Detector de Carros', np.squeeze(detect.render()))

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()


