import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
         "Area":[1000,2000,3000],
         "Score" :[9,8,7],
         "Bedroom" : [2,3,4],
         "Price" : [56,60,66]

        }

df = pd.DataFrame(data)

x = df[["Area","Score","Bedroom"]]
y = df[["Price"]]

model = LinearRegression()
model.fit(x,y)

new_house = pd.DataFrame([[4000,6,5]],columns = ["Area","Score","Bedroom"])
prediction = model.predict(new_house)
print("Predicted scored is",prediction)
