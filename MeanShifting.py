import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanShifting:
	centroid_list = []

	def __init__(self, data):
		self.data = data

	def train(self):
		x_list = []
		y_list = []
		points_list = []
		for i in range(20):
			for j in range(20):
				vector = []
				for k in self.data.columns:
					vector.append(((self.data[k].max() - self.data[k].min()) * (i if k == 'x' else j) / 20) + self.data[k].min())
				points_list.append(vector)
				self.centroid_list.append(Centroid([vector[0], vector[1]], 1/20))
		points_list = np.array(points_list).T
		plt.scatter(points_list[0], points_list[1], s=20)
		plt.scatter(self.data[self.data.columns[0]], self.data[self.data.columns[1]], c='r', s=1)
		plt.show()

		cent = []
		for i in range(200):
			self.step()
			cent = [[i.position[0], i.position[1]] for i in self.centroid_list]
			print("iteration " + str(i))
		cent = pd.DataFrame(cent, columns=list('xy'))
		print("centroid count: " + str(len(cent)))
		plt.scatter(cent['x'], cent['y'], s=20)
		plt.scatter(self.data['x'], self.data['y'], c='r', s=2)
		plt.show()

	def step(self):
		updated_centroids = []
		for i in self.centroid_list:
			self.update_centroid(i)
			append = True
			for j in updated_centroids:
				if j.position[0] == i.position[0] or j.position[1] == i.position[1]:
					append = False
			if append:
				updated_centroids.append(i)
		self.centroid_list = updated_centroids

	def update_centroid(self, c):
		in_range_x = self.data[c.position[0] - c.radius < self.data['x']]
		in_range_x = in_range_x[c.position[0] + c.radius > in_range_x['x']]['x']
		in_range_y = self.data[c.position[1] - c.radius < self.data['y']]
		in_range_y = in_range_y[c.position[1] + c.radius > in_range_y['y']]['y']
		c.update_position([in_range_x.mean(), in_range_y.mean()])


class Centroid:
	def __init__(self, position, radius):
		self.position = position
		self.radius = radius

	def update_position(self, mean_vector):
		for i in range(len(mean_vector)):
			self.position[i] = mean_vector[i]

