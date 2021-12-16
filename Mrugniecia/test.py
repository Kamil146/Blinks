import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--program", required=True, help="Wybór pliku programu")
parser.add_argument("-t", "--test", required=True, help="Wybór pliku testu")

args = parser.parse_args()

with open(f'Wyniki testów/{args.test}', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open(f'Wyniki testów/{args.program}', newline='') as csvfile:
    data2 = list(csv.reader(csvfile))
test=[1,2,3]
TP=0
TN=0
FP=0
FN=0
precision=0
recall=0
F1=0
data = np.asarray(data)
data2 = np.asarray(data2)

plt.plot(data[:,0], label = "test")
plt.plot(data2[:,0], label = "program")


for i, dat2 in enumerate(data2):
    if int(data2[i][0])==0:
        if(data2[i][0]==data[i][0]):
            TN+=1
        else:
            FN+=1
    elif int(data2[i][0])==1:
        if(data2[i][0]==data[i][0]):
            TP+=1
        else:
            FP+=1


print(TN)
print(TP)
print(FN)
print(FP)

precision=(TP/(TP+FP))
recall =(TP/(TP+FN))
F1=(2*precision*recall)/(precision+recall)

print(precision)
print(recall)
print(F1)


plt.legend()
plt.show()


print(len(data))
print(len(data2))





