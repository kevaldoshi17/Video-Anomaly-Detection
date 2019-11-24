import numpy as np
from scipy.io import loadmat,savemat
import os
import matplotlib.pyplot as plt

Predictions = os.listdir("Allmat/")

def perf_measure(y_actual, y_pred):
	TP = 0
	FP = 0
	TN = 0
	FN = 0

	for i in range(0,len(y_pred)): 
		# print(i)
		if y_actual[i]==y_pred[i]==1:
		   TP += 1
		if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
		   FP += 1
		if y_actual[i]==y_pred[i]==0:
		   TN += 1
		if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
		   FN += 1

	return(TP, FP, TN, FN)


TPR = list()
FPR = list()
for thr in range(1,9000,50):
	print(thr)
	thr = thr*0.00005
	GTP = 0
	GFP = 0
	GTN = 0
	GFN = 0

	for prediction in Predictions:
		GT = loadmat("Allmat/" + prediction)['truth'][0]
		NT = loadmat("../Eval_Res/" + prediction)['prediction']
		FT = np.zeros((1,len(GT)))
		
		d_len = int(len(GT)/32)
		
		for i in range(len(NT)):
			FT[0,1*d_len*(i-1):d_len*i] = NT[i]
			if i == len(NT)-1:
				FT[0,1*d_len*(i-1):-1] = NT[i]
		# print((FT.shape[1]),len(GT))
		for i in range(0,FT.shape[1]):
			if FT[0,i] >= thr:
				FT[0,i] = 1
			else:
				FT[0,i] = 0
		# print(GT,FT)
		TP,FP,TN,FN = perf_measure(GT,FT[0,:])
		GTP+=TP
		GFP+=FP
		GTN+=TN
		GFN+=FN

	TPR.append(GTP/(GTP+GFN))
	FPR.append(GFP/(GFP+GTN))

print(TPR,FPR)

plt.plot(np.array(FPR),np.array(TPR))   
plt.show()






