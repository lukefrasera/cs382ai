import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3,14],'ro-')

# set your ticks manually
ax.xaxis.set_ticks([1.,2.,3.,10.])
ax.grid(True)

plt.show()