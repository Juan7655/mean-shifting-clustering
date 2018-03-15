import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanShifting:
	centroid_list, factor_list, init_centroid_num = [], [], 20

	def __init__(self, data):
		self.data = data
		self.min = np.mat([self.data[k].min() for k in self.data.columns]).T
		self.max = np.mat([self.data[k].max() for k in self.data.columns]).T
		self.range = np.array(self.max - self.min)
		self.factor_list = np.zeros([len(self.data.columns), 1])

	def train(self, max_iter=1000, show_graphs=True):
		while True:  # creates initial uniform matrix on centroids
			temp_list = np.array((self.factor_list * self.range / self.init_centroid_num) + self.min).T[0]
			self.centroid_list.append(Centroid(temp_list, radius=(1/400) * self.init_centroid_num))
			if self.next_list() == -1:  # the cycle has returned to initial point
				break  # reached maximum number of initial clusters. Stop.
		self.draw_graph(show_graphs)  # show initial matrix of centroids with raw unclassified dataset

		for i in range(max_iter):  # step until maximum iterations number is reached, or convergence state
			print("iteration {0}/{1}".format(i, max_iter))
			if self.step() == 0:  # convergence state reached
				print("Convergence reached!")
				break
		print("centroid count: {0}".format(len(self.centroid_list)))

		self.draw_graph(show_graphs)  # show final graph with centroids positions

	def draw_graph(self, show_graph=False):
		if len(self.data.columns) == 2 and show_graph:  # do not attempt to plot if dataset is not 2-Dimensional
			cent = pd.DataFrame([[i.position[j] for j in range(len(i.position))] for i in self.centroid_list])
			# it may be assumed that there are only 2 dimensions
			plt.scatter(cent[cent.columns[0]], cent[cent.columns[1]], s=20, zorder=2)
			plt.scatter(self.data[self.data.columns[0]], self.data[self.data.columns[1]], c='r', s=2)
			plt.xlabel(self.data.columns[0])
			plt.ylabel(self.data.columns[1])
			plt.show()

	def step(self):
		updated_centroids = []
		distance = 0
		for i in self.centroid_list:
			distance += self.update_centroid(i)
			append = not np.isnan(i.position).any()
			for j in updated_centroids:
				if get_distance(i.position, j.position, n=2) < 0.1 or not i.append:
					append = False
					break
			if append:
				updated_centroids.append(i)
		self.centroid_list = updated_centroids
		return distance

	def update_centroid(self, c):
		in_range = self.data
		for i in range(len(self.data.columns)):
			in_range = in_range[(c.position[i] - c.radius < in_range[in_range.columns[i]]) &
			                    (c.position[i] + c.radius > in_range[in_range.columns[i]])]
		res = c.update_position([in_range[i].mean() for i in in_range.columns])

		in_range = self.data
		for i in range(len(self.data.columns)):
			in_range = in_range[(c.position[i] - c.radius < in_range[in_range.columns[i]]) &
			                    (c.position[i] + c.radius > in_range[in_range.columns[i]])]
		c.append = len(in_range) > 40
		return res

	def next_list(self):
		max_i = len(self.factor_list) - 1
		for i in range(max_i + 1):
			self.factor_list[max_i - i] += 1
			if self.factor_list[max_i - i] != self.init_centroid_num:
				return
			self.factor_list[max_i - i] = 0
		if self.factor_list[max_i] == 0:
			return -1


def get_distance(p, q, n=2):
	diff = (np.abs(p - q) ** n).T
	try:
		diff.columns = [0]
		suma = np.sum(diff)
		result = suma[0] ** (1 / n)
	except AttributeError:
		result = np.sum(diff) ** (1 / n)
	return result


class Centroid:
	append = True

	def __init__(self, position, radius):
		self.position = position
		self.radius = radius

	def update_position(self, mean_vector):
		distance = 0
		for i in range(len(mean_vector)):
			distance += get_distance(self.position[i], mean_vector[i])
			self.position[i] = mean_vector[i]
		return distance
