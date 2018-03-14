import pandas as pd
import numpy as np
import MeanShifting as Shift
import datetime


def run():
	now = datetime.datetime.now()
	data = pd.read_csv("data2.csv")
	data.columns = list('xy')
	data['x'] = (np.array(data['x']) - data['x'].min()) / (data['x'].max() - data['x'].min())
	data['y'] = (np.array(data['y']) - data['y'].min()) / (data['y'].max() - data['y'].min())
	model = Shift.MeanShifting(data)
	model.train(show_graphs=True)
	print("Execution time: {0}".format(datetime.datetime.now() - now))


if __name__ == "__main__":
	run()
