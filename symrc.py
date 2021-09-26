from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np
graph = [
	[1, 1, 0, 0, 1],
	[1, 1, 0, 0, 0],
	[0, 0, 1, 0, 0],
	[0, 0, 0, 1, 1],
	[1, 0, 0, 1, 0],
]

graph_csr = csr_matrix(graph)

print(graph)

p = reverse_cuthill_mckee(graph_csr,True)
print(p)
I = np.eye(5,5)
P = I[p]
print(P)
print(np.matmul(np.matmul(P,graph),P.T))
#import numpy as np

#x=np.arange(32).reshape((8,4))
#print (x[[4,2,1,7]])