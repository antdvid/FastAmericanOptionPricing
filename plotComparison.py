import numpy as np
import matplotlib.pyplot as plt
import re

method_name = ['LSMC', 'Trinomial', 'PDE', 'Spectral', 'Bjerksund Stensland']
file_name = ['MonteCarlo', 'TrinomialAmerican', 'PDEAmerican', 'Spectral', 'BjerksundStensland']

for m in file_name:
    fname = 'data/' + m + '.txt'
    #clean data
    f = open(fname, 'r')
    res = f.read().split('\n')
    #remove anything except number, white space and dot
    res = np.array([re.sub(r'([^\s\w.E-]|_)+', '', s).split(' ') for s in res])
    time = [float(r[0]) for r in res[0: -1]]
    err = [float(r[1]) for r in res[0: -1]]
    plt.loglog(time, err, 'o-')

plt.ylim([1e-8, 1])
plt.xlim([1e-3, 1e5])
plt.legend(method_name, loc="lower right")
plt.xlabel('avergage time [ms]')
plt.ylabel('error')
plt.show()