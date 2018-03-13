import pandas as pd
import MeanShifting as Shift
import datetime


def run():
	now = datetime.datetime.now()
	data = pd.read_csv("data.csv")
	model = Shift.MeanShifting(data)
	model.train(show_graphs=True)
	print("Execution time: " + str(datetime.datetime.now() - now))


if __name__ == "__main__":
	run()
