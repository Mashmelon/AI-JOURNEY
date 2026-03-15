import pandas as pd

data = {
         "Product" : ["Laptop","Phone","Headset","Phone","Laptop"],
         "Quantity" : [6,7,8,9,10],
         "Price" : [800,500,300,800,500]
       }
df = pd.DataFrame(data)

df["Total_Revenue"] = df["Quantity"] *df["Price"]
Total_Sales = print("Total Sales : ",df["Total_Revenue"].sum())

Best_seller = df.groupby("Product")["Quantity"].sum()
print("Best seller : ",Best_seller.idxmax())
Best_product = df.groupby("Product")["Total_Revenue"].sum()
print(Best_product.to_string())
print("Best product : ",Best_product.idxmax())




         
        
                      
                    
