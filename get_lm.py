import numpy as np
import numpy.matlib
import time
from numba import jit
from joblib import Parallel, delayed
# >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))


F = np.load('Features.npy')

Ng = 5000
Mg = 19000



F = F.reshape(4096,-1)

X_N = F[:,0:Ng]

X_M = F[:,Ng:Ng+Mg]
er = np.zeros((1,Ng))
print(X_N.shape)

def knndis(t,X_M,Mg,i):
	# tart = time.time()
	print(i)
	dist = np.sum((np.transpose(np.matlib.repmat(t,Mg,1)) - X_M)**2,0) 
	# print(dist,len(dist))
	dist = np.sort(dist)
	return dist[0]

# for i in range(0,Ng):
	
# 	t = X_N[:,i]
# 	er[0,i] = knndis(t,X_M,Mg)[0]
# 	print(i)


er = Parallel(n_jobs=-1)(delayed(knndis)(X_N[:,i],X_M,Mg,i) for i in range(Ng))


np.save('errors.npy',er)

print(er)
