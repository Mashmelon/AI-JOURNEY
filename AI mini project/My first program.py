
import numpy as np
from sklearn.linear_model import LinearRegression

Hours = np.array([[1],[2],[3],[4]])
Scores = np.array([30,40,50])

model = LinearRegression()
model.fit(Hours,Scores)



pred = model.predict([8])
print(f"If a student studied for {Hours} he/she can score about {pred[0]:.2f}")
print("Model trained successfully")
