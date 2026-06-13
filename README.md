# Handson-L8-Spark-Streaming-MachineLearning-MLlib
### Task 4
Task 4 consisted of two main parts: offline model trainning, and real-time inference. More specifically:
1. **Offline Model Trainning:** A LinearRegression model is trained using a static CSV file (training-dataset.csv). The model learns the relationship between `distance_km` (the feature) and `fare_amount` (the label). The trained model is then saved to disk.
3. **Real-Time Inference:** The script ingests live ride data from the socket. For each incoming ride, it uses the pre-trained model to predict the fare based on the trip's
distance. It then calculates the deviation between the actual fare and the predicted fare to identify potential anomalies.
