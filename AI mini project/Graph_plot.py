import matplotlib.pyplot as plt

days = [1,2,3,4,5]
sales = [100,150,200,180,250]

plt.plot(days, sales)
plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales Trend")
plt.show()
plt.scatter(days, sales)
plt.show()
plt.plot(days, model.predict(sales))
plt.show()
