import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

data = Table.read('/volumes/ssd/BSH_data/250820/sate_0_1.csv', format='ascii')

x = np.linspace(0,len(data['col2'])-1, len(data['col2']))
plt.title('No satelite')
plt.scatter(x, data['col2'], s=3)
plt.xlabel('Frame')
plt.ylabel('satelite num')
plt.show()

