import matplotlib.pyplot as plt


plt.figure(figsize=(12,12))
plt.plot([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9], [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2], 'r+',markersize=4)
plt.plot([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9], [3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4], 'bo',markersize=4)
plt.axis([0,100,0,100])
plt.grid()
plt.show()