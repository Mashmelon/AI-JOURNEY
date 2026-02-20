import pandas as pd

data = {
    "Product": ["Laptop","Phone","Tablet","Laptop","Phone"],
    "Price": [800,500,300,800,500],
    "Quantity": [2,3,4,1,2]
}

df = pd.DataFrame(data)

df["Total_Sale"] = df["Price"] * df["Quantity"]

print("Total Revenue:", df["Total_Sale"].sum())

sales_quantity = df.groupby("Product")["Quantity"].sum()
print(sales_quantity.to_string())
print("Best selling product:", sales_quantity.idxmax())
Sales = df.groupby("Product")["Total_Sale"].sum()


print("Highest revenue product:",Sales.idxmax())
      

print("Average Sale:", df["Total_Sale"].mean())
