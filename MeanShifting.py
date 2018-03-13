import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanShifting:
	centroid_list = []
	factor_list = []

	def __init__(self, data):
		self.data = data
		self.min = np.mat([self.data[k].min() for k in self.data.columns]).T
		self.max = np.mat([self.data[k].max() for k in self.data.columns]).T
		self.range = np.array(self.max - self.min)
		self.factor_list = np.zeros([len(self.data.columns), 1])

	def train(self, show_graphs=True):
		points_list = []
		temp_list2 = []

		while True:
			temp_list = np.array((self.factor_list * self.range / 20) + self.min).T
			self.centroid_list.append(Centroid(temp_list[0], radius=1/20))
			points_list.append(temp_list.T)
			temp_list2.append(self.factor_list)
			if self.next_list() == -1:
				break
		points_list = np.array(points_list).T
		if len(self.factor_list) == 2 and show_graphs:
			plt.scatter(points_list[0][0], points_list[0][1], s=20)
			plt.scatter(self.data[self.data.columns[0]], self.data[self.data.columns[1]], c='r', s=1)
			plt.show()

		for i in range(200):
			self.step()
			print("iteration " + str(i) + "/" + str(199))
		print("centroid count: " + str(len(self.centroid_list)))

		if len(self.factor_list) == 2 and show_graphs:
			cent = pd.DataFrame([[i.position[0], i.position[1]] for i in self.centroid_list], columns=list('xy'))
			plt.scatter(cent['x'], cent['y'], s=20)
			plt.scatter(self.data['x'], self.data['y'], c='r', s=2)
			plt.show()

	def step(self):
		updated_centroids = []
		for i in self.centroid_list:
			self.update_centroid(i)
			append = True
			for j in updated_centroids:
				if self.get_distance(i.position, j.position, n=2) < 0.3:
					# if j.position[0] == i.position[0] or j.position[1] == i.position[1]:
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

	def next_list(self):
		max_i = len(self.factor_list) - 1
		for i in range(max_i + 1):
			self.factor_list[max_i - i] += 1
			if self.factor_list[max_i - i] != 20:
				return
			self.factor_list[max_i - i] = 0
		if self.factor_list[max_i] == 0:
			return -1

	@staticmethod
	def get_distance(p, q, n=8):
		diff = (np.abs(p - q) ** n).T
		try:
			diff.columns = [0]
			suma = np.sum(diff)
			result = suma[0] ** (1 / n)
		except AttributeError:
			result = np.sum(diff) ** (1 / n)
		return result


class Centroid:
	def __init__(self, position, radius):
		self.position = position
		self.radius = radius

	def update_position(self, mean_vector):
		for i in range(len(mean_vector)):
			self.position[i] = mean_vector[i]
