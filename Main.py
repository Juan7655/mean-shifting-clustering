import pandas as pd
import MeanShifting as shift


def run():
	data = pd.read_csv("data.csv")
	model = shift.MeanShifting(data)
	model.train()


if __name__ == "__main__":
	run()
