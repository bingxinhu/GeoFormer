import matplotlib.pyplot as plt

slopes = [0.1287, 0.2592, 0.4224, 0.8303, -0.9017, -1.0857, 1.1075, 2.1158, -3.0925, 4.2077]
distances = [131.0936, 241.1351, 254.0848, 186.2207, 296.1186, 198.2965, 183.6856, 117.0056, 122.8338, 189.3173]

plt.scatter(slopes, distances)
plt.xlabel('Slope')
plt.ylabel('Distance')
plt.title('Slope vs Distance')
plt.show()