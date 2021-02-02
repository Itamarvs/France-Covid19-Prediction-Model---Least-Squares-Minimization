import csv
import numpy as np
import matplotlib.pyplot as plt

# create a dictionary of the new cases/population density for each country in the model
Countries = {"ESP": [], "FRA": [], "GBR": [], "DEU": [], "CHE": [], "ITA": [], "PRT": [], "BEL": []}

# read the data from the csv file and put the first 80% of dates into the train model.
# normalize data by a moving average by week
with open('Covid19_data.csv') as file:
	data = list(csv.reader(file))
	interval = 7
	for i in range(interval, len(data)):
		currCountry = data[i][0]
		avgLast5DaysSick = [0]
		currCountryDens = float(data[i][5])
		currCountryPop = float(data[i][4])
		if str(data[i][0]) != str(data[i-interval][0]):
			continue
		for j in range(interval):
			avgLast5DaysSick[0] += float(data[i - j][3])
		avgLast5DaysSick[0] = (avgLast5DaysSick[0] / interval)
		Countries[currCountry].append(avgLast5DaysSick)

# read France's new cases into a column vector b
b = np.asarray(Countries["FRA"][:round(0.8 * len(Countries["FRA"]))])  # 80% data for training the model

# b vector of the last 20% of data (France), for testing the model
b_test = np.asarray(Countries["FRA"][len(b):])

# # create the data matrix A for training the model (based on 80% of the data)
# # create the data matrix ATest for testing the model (based on the last 20% of the data)
tempA = []
tempATest = []
for day in range(round(len(Countries["FRA"]))):
	if day < round(0.8*len(Countries["FRA"])):
		currDayData = []
		for country in Countries.keys():
			if country not in ("PRT", "CHE", "BEL"):
			# if country in ("FRA"):
				continue
			currDayData.append(Countries[country][day][0])
		tempA.append(currDayData)
	else:
		currDayDataTest = []
		for country in Countries.keys():
			if country not in ("PRT", "CHE", "BEL"):
			# if country in ("FRA"):
				continue
			currDayDataTest.append(Countries[country][day][0])
		tempATest.append(currDayDataTest)

# A is the data matrix for the model training
A = np.asarray(tempA)
# ATest is the data matrix for the model testing (tries to predict b_test based on ATest data)
ATest = np.asarray(tempATest)

# Use weighted least squares. give more weight to the middle part of the data
W = np.eye(len(b))
for k in range(40,150):
	W[k][k] = 10

x = np.linalg.inv(np.transpose(A) @ W @ A) @ np.transpose(A) @ W @ b
r = A @ x - b
# calculate the train MSE
trainMSE = np.linalg.norm(r)

predictedCases = ATest @ x
# calculate the test MSE
testMSE = np.linalg.norm(predictedCases - b_test)
print("train MSE is: \n", trainMSE, '\n\n', 'test MSE is:\n', testMSE, '\n\n', 'x is: \n', x)

#plot the model
plt.title('Model Training')
plt.xlabel('Day')
plt.ylabel('New Cases')
xAxis = np.arange(0, len(b))
plt.plot(xAxis, b, 'maroon')
plt.plot(xAxis, A @ x, 'lime')
plt.legend(['Actual new cases per day (normalized) in france', 'Model new cases per day (normalized) in france'])
plt.show()

plt.title('Model Prediction')
plt.xlabel('Day')
plt.ylabel('New Cases')
xAxis = np.arange(0, len(b_test))
plt.plot(xAxis, b_test, 'royalblue')
plt.plot(xAxis, predictedCases, 'gold')
plt.legend(['Actual new cases per day (normalized) in france', 'Model new cases per day (normalized) in france'])
plt.show()