# Handson-L8-Spark-Streaming-MachineLearning-MLlib
### Task 4
Task 4 consisted of two main parts: offline model trainning, and real-time inference. More specifically:
1. **Offline Model Trainning:** A LinearRegression model is trained using a static CSV file (training-dataset.csv). The model learns the relationship between `distance_km` (the feature) and `fare_amount` (the label). The trained model is then saved to disk.
3. **Real-Time Inference:** The script ingests live ride data from the socket. For each incoming ride, it uses the pre-trained model to predict the fare based on the trip's
distance. It then calculates the deviation between the actual fare and the predicted fare to identify potential anomalies.

Output files from the execution of Task 4 are stored under the directory `models/fare_model`. The Python script `task4.py` contains the code utized for Task 4.

Furthermore, here are two images with screeshots taken during the execution of the script `task4.py`:

![hello text](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task4-1.png)

![hello text](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task4-2.png)
