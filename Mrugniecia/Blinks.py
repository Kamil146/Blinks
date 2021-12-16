import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

import argparse
import csv
from configparser import ConfigParser


#distance() - obliczanie odległości euklidesowej
# @param lx[in] - współrzędna x pierwszego punktu
# @param ly[in] - współrzędna y pierwszego punktu
# @param rx[in] - współrzędna x drugiego punktu
# @param ry[in] - współrzędna y drugiego punktu
# @param b[out] - odleglość między punktami
def distance(lx,ly, rx, ry):
    b = np.sqrt((lx-rx)**2 +(ly-ry)**2)
    return b

#funkcja pusta, którą wywołujemy podczas tworzenia suwaków
def nothing(x):

    pass

#sp() - obliczanie spadków
# @param m1[in] - wartość proporcji oka z poprzedniej klatki
# @param m2[in] - wartość proporcji oka w aktualnej klatce
# @param last[in] - wartość ostatniego spadku
# @param suma[in] - wartość ostatniej sumy spadków
# @param suma[out] - wartość nowej sumy spadków
# @param spadek[out] - wartość aktualnego spadku
def sp(m1,m2,last,suma):

    if(m1!=0):

        spadek = ((m1 - m2) / m1)

        #jesli w tej i poprzedniej klatce były spadki to sumujemy
        if(spadek>0 and last>0):
            suma += spadek

            return suma,spadek
        # jeśli w tej i poprzedniej klatce były wzrosty to sumujemy
        elif(spadek<0 and last<0):
            suma += spadek


            return suma,spadek
        #jeśli jest to pierwszy wzrost/spadek
        else:
            suma=spadek
            return suma,spadek
    else:
        return 0,0

#Obliczanie stosunku proporcji oka
# @param fl[in] - wszystkie punkty charakterystyczne twarzy
# @param peye[in] - punkty charakterystyczne oka w predyktorze
# @param ratio[out] - stosunek proporcji oka
def ratio(fl,peye):
#górne punkty oka
    top1 = [fl.part(peye[1]).x, fl.part(peye[1]).y]
    top2 = [fl.part(peye[2]).x, fl.part(peye[2]).y]
#dolne punkty oka
    bot2 = [fl.part(peye[4]).x, fl.part(peye[4]).y]
    bot1 = [fl.part(peye[5]).x, fl.part(peye[5]).y]
#lewy punkt oka
    left = [fl.part(peye[0]).x, fl.part(peye[0]).y]
#prawy punkt oka
    right = [fl.part(peye[3]).x, fl.part(peye[3]).y]
#rysowanie punktów
    if(type=="points"):
        point1 = cv2.circle(frame, top1, radius=0, color=(0, 0, 255), thickness=-1)
        point1 = cv2.circle(frame, top2, radius=0, color=(0, 0, 255), thickness=-1)
        point1 = cv2.circle(frame, bot1, radius=0, color=(0, 0, 255), thickness=-1)
        point1 = cv2.circle(frame, bot2, radius=0, color=(0, 0, 255), thickness=-1)
        point1 = cv2.circle(frame, left, radius=0, color=(0, 0, 255), thickness=-1)
        point1 = cv2.circle(frame, right, radius=0, color=(0, 0, 255), thickness=-1)
    if(type=="lines"):
        hor_line = cv2.line(frame, (left[0], left[1]), (right[0], right[1]), (0, 0,255), 1)
        ver_line1 = cv2.line(frame, (top1[0], top1[1]),(bot1[0], bot2[1]), (0, 0, 255), 1)
        ver_line2 = cv2.line(frame, (top2[0], top2[1]),(bot2[0], bot2[1]), (0, 0, 255), 1)
 #liczenie odległości pomiedzy punktami
    horizontal = distance(left[0], left[1], right[0], right[1])
    vertical1 = distance(top1[0], top1[1], bot1[0], bot1[1])
    vertical2 = distance(top2[0], top2[1], bot2[0], bot2[1])
 #obliczanie stosunku
    ratio = (vertical1+vertical2) / (2.0*horizontal)
    return ratio

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video", default="0", help="Wybór pliku, domyślnie kamera")
parser.add_argument("-d", "--detector",default="d", help="Wybór detektora twarzy: 'h' kaskada haara, 'd' hog z dlib")
parser.add_argument("-s", "--scale", default="100", help="wybór skali")
parser.add_argument("-a", "--algorithm", default="0", help="wybór algoyrtmu - 0 proporcje , 1 przyrost/spadek")
parser.add_argument("-w", "--write",default="0", help="zapis wyników do pliku csv")
args = parser.parse_args()

config = ConfigParser()
config.read("config.ini")
frames=config.getint('parameters','frames_number')
if(args.algorithm=="0"):
    threshold=config.getfloat('parameters','threshold')
elif(args.algorithm=="1"):
    threshold = config.getfloat('parameters','threshold2')
type=config.get('parameters','eyeshow_type')
plot_show=config.getint('parameters','plot_show')
hist = config.getint('preprocessing','histogram')
filter = config.get('preprocessing','filter_type')
kernel = config.getint('preprocessing','kernel_size')
haar = config.get('files','haar_classificator')

#detekcja przodu twarzy
if(args.detector[0]=="d"):
    detector = dlib.get_frontal_face_detector()
elif(args.detector[0]=="h"):

    detector = cv2.CascadeClassifier(haar)

#detekcja punktów charakterystycznych twarzy
predictor = dlib.shape_predictor(config.get('files','predictor'))

blinks_plot=[]
proportions_plot=[]
fall_plot=[]
blinktime_plot = []
counter=0
counter2=0
blinks=0
proportions=0
fall=0
last = 0
first = True
flag1=0
flag2=0
blink_time=0
#punkty dla oczu
rpoints=[42, 43, 44, 45, 46, 47]
lpoints=[36, 37, 38, 39, 40, 41]

if(args.video=="0"):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(f"Pliki testowe/{args.video}")

cv2.namedWindow('Frame')
cv2.createTrackbar('Granica','Frame',200,600,nothing)
cv2.createTrackbar('Klatki','Frame',1,10,nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Frame',0,1,nothing)

while True:
    ret,frame=cap.read()
    if ret == False:
        break
    scale =int(args.scale)
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)

    frame = cv2.resize(frame, dim)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # zaaplikowanie adaptywnego wyrównania histogramu
    if(hist==1):
       #clipLimit - podajemy granicę limitowania kontrastu, domyślnie 40
       clahe = cv2.createCLAHE(clipLimit=2)
       equalized = clahe.apply(gray)
    #zastosowanie zwykłego wyrównania histogramu
    elif(hist==0):
        equalized = cv2.equalizeHist(gray)
    # filtr gaussa
    if(filter=="g"):
        imgBlur = cv2.GaussianBlur(equalized, (kernel,kernel),1)
        img = imgBlur

    # filtr medianowy
    elif(filter=="m"):
        median = cv2.medianBlur(equalized, kernel)
        img = median

    #detekcja przodu twarzy
    if (args.detector[0] == "d"):
        faces = detector(img)
    elif (args.detector[0] == "h"):

        #  skalowanie 1.05-1.4 im większy tym szybciej ale mniej dokładnie
        try:
            faces = detector.detectMultiScale(gray, float(args.detektor[1]), int(args.detektor[2]))
        except IndexError:
            faces = detector.detectMultiScale(gray, 1.2, 5)

    for face  in faces:

        if (args.detector[0] == "h"):
            #zamiana typu danych otrzymanych z kaskady Haara na typ rec
            face= dlib.rectangle(int(face[0]), int(face[1]), int(face[0] + face[2]), int(face[1] + face[3]))

        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        #wyszukiwania punktów charakterystycznych twarzy
        landmarks = predictor(img,face)
        #proporcje dla prawego oka
        r_ratio = ratio(landmarks, rpoints)
        #proporcje dla lewego oka
        l_ratio = ratio(landmarks, lpoints)
        proportions_last = proportions
        proportions = (l_ratio + r_ratio) / 2
        fall , last = sp(proportions_last, proportions,last,fall)

        #algorytm dla progu
        if(args.algorithm=="0"):
            if proportions < threshold:
                counter += 1
                blink_time=1
            else:
                if counter >= frames and counter<=30 :
                    blinks += 1
                    blinktime_plot.extend([1]*counter)
                    counter2 = counter
                else:
                    blinktime_plot.extend([0] * counter)
                blink_time = 0
                counter = 0

        #algorytm dla spadków i wzrostów
        elif(args.algorithm=="1"):

            #zakładany początek mrugniecia gdy suma spadków przekroczy granice
            if fall > threshold or flag1==1:
                counter +=1
                blink_time = 1
                flag1 = 1
                #zakładany koniec mrugnięcia gdy suma wzrostów przekroczy granice
                if -fall > threshold:
                    #zmniejszenie licznika o 1 gdyż w tej klatce zakładamy już oko otwarte
                    counter-=1
                    #sprawdzenie czy liczba klatek większa od minimalnie wymaganej
                    if counter>= frames:
                        blinks+=1
                        blinktime_plot.extend([1] * (counter))

                    else:
                        blinktime_plot.extend([0] * (counter))
                    counter2 = counter
                    counter = 0
                    flag1 = 0
                    blink_time = 0
                #gdy nie nastapi otwarcie oka w czasie
                elif counter>=30:
                    blinktime_plot.extend([0] * (counter))
                    counter = 0
                    counter2 = counter
                    flag1 = 0
            else:
                counter=0
                blink_time=0

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, "Blinks: "+str(blinks), (0,int(height*0.1)),cv2.FONT_HERSHEY_PLAIN , 1.5, (0, 0, 255), 2)
    cv2.putText(frame, "Ratio: {:0.2f}".format(proportions), (0,int(height*0.2)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Frames: "+str(counter2), (0,int(height*0.3)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    #tworzenie wykresów
    if(blink_time==0):

        blinktime_plot.append(blink_time)

    proportions_plot.append(proportions)
    blinks_plot.append(blinks)
    fall_plot.append(fall)

    plt.subplot(2, 2, 3)
    plt.plot(proportions_plot)
    plt.title("Proporcje oka")
    plt.subplot(2, 2, 1)
    plt.plot(blinks_plot)
    plt.title("Suma mrugnięć")
    plt.subplot(2, 2, 2)
    plt.plot(fall_plot)
    plt.title("Spadki i wzrosty")
    plt.subplot(2, 2, 4)
    plt.plot(blinktime_plot)
    plt.title("Czas zliczony mrugnięć")

    cv2.imshow("Frame",frame)

    t = cv2.getTrackbarPos('Granica', 'Frame')
    f = cv2.getTrackbarPos('Klatki', 'Frame')
    s = cv2.getTrackbarPos(switch, 'Frame')
    if s == 0:
        frames = config.getint('parameters','frames_number')
        if (args.algorithm == "0"):
            threshold = config.getfloat('parameters', 'threshold')
        elif (args.algorithm == "1"):
            threshold = config.getfloat('parameters', 'threshold2')

    else:
        threshold = t / 1000
        frames = f

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        plt.show()

if(plot_show==1):
    plt.show()

if(args.write!="0"):
    with open(f'Wyniki testów/{args.write}', 'w',newline="", encoding='UTF8') as f:

        w = csv.writer(f)

        #zapisanie w kolumnie do pliku
        for word in blinktime_plot:
            w.writerow([word])

cv2.destroyAllWindows()
