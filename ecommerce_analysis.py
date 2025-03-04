import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    print("Scikit-learn imports successful!")
except ImportError as e:
    print(f"Error importing scikit-learn modules: {e}")
    exit()

# Step 1: Data Collection and Cleaning
orders = pd.read_csv("olist_orders_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")

print("Orders Dataset Preview:")
print(orders.head())
print("\nCustomers Dataset Preview:")
print(customers.head())
print("\nOrders Info:")
print(orders.info())

orders = orders[orders['order_status'] == 'delivered']
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['estimated_time'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
orders.dropna(subset=['delivery_time', 'estimated_time'], inplace=True)
data = orders.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

print("Cleaned Data Preview:")
print(data[['order_id', 'delivery_time', 'estimated_time', 'customer_state']].head())  # Fixed typo here

# Export cleaned data for Power BI
data.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved as 'cleaned_data.csv'")

db_params = {
    'dbname': 'ecommerce_db',
    'user': 'postgres',
    'password': 'password',  
    'host': 'localhost',
    'port': '5432'
}
engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")
data.to_sql('orders', engine, if_exists='replace', index=False)
print("Data successfully loaded into PostgreSQL!")

# Step 2: Exploratory Data Analysis (EDA)
data = pd.read_sql("SELECT * FROM orders", engine)

print("Delivery Time Statistics:")
print(data['delivery_time'].describe())

state_delivery = data.groupby('customer_state')['delivery_time'].agg(['mean', 'count']).sort_values('mean', ascending=False)
state_delivery.to_csv('state_delivery.csv', index=True)
print("\nAverage Delivery Time by State:")
print(state_delivery.head())

plt.figure(figsize=(10, 6))
sns.histplot(data['delivery_time'], bins=30, kde=True)
plt.title('Distribution of Delivery Times')
plt.xlabel('Delivery Time (days)')
plt.ylabel('Frequency')
plt.savefig('delivery_time_histogram.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='customer_state', y='delivery_time', data=data)
plt.xticks(rotation=90)
plt.title('Delivery Time by State')
plt.xlabel('State')
plt.ylabel('Delivery Time (days)')
plt.savefig('delivery_time_boxplot.png')
plt.show()

anomalies = pd.read_sql("""
    SELECT order_id, customer_state, delivery_time
    FROM orders
    WHERE delivery_time > 30
    ORDER BY delivery_time DESC
    LIMIT 10
""", engine)
anomalies.to_csv('anomalies.csv', index=False)
print("\nTop 10 Longest Deliveries:")
print(anomalies)

# Step 3: Data Modeling and Analysis
X = pd.get_dummies(data[['customer_state']]).join(data['estimated_time'])
y = data['delivery_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance (Random Forest):")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Export predictions for Power BI
predictions = pd.DataFrame({'Actual_Delivery_Time': y_test, 'Predicted_Delivery_Time': y_pred})
predictions.to_csv('predictions.csv', index=False)
print("Predictions saved as 'predictions.csv'")