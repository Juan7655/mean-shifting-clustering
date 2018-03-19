import pandas as pd
import numpy as np
import MeanShifting as Shift
import datetime


def run():
	now = datetime.datetime.now()
	data = pd.read_csv("data3.csv")
	# data = pd.read_csv("results4-feat.csv")

	# data normalization
	for i in data.columns:
		data[i] = (np.array(data[i]) - data[i].min()) / (data[i].max() - data[i].min())

	model = Shift.MeanShifting(data)
	model.train(max_iter=200, show_graphs=True)
	print("Execution time: {0}".format(datetime.datetime.now() - now))


if __name__ == "__main__":
	run()
