# Handson-L8-Spark-Streaming-MachineLearning-MLlib
### Task 4
Task 4 consisted of two main parts: offline model trainning, and real-time inference. More specifically:
1. **Offline Model Trainning:** A LinearRegression model is trained using a static CSV file (training-dataset.csv). The model learns the relationship between `distance_km` (the feature) and `fare_amount` (the label). The trained model is then saved to disk.
3. **Real-Time Inference:** The script ingests live ride data from the socket. For each incoming ride, it uses the pre-trained model to predict the fare based on the trip's
distance. It then calculates the deviation between the actual fare and the predicted fare to identify potential anomalies.

Output files from the execution of Task 4 are stored under the directory `models/fare_model`. The Python script `task4.py` contains the code utilized for Task 4.

Furthermore, here are two images with screeshots taken during the execution of the script `task4.py`:

![Output from task 4](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task4-1.png)

![Screenshot from task 4](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task4-2.png)

### Task 5
Task 5 also involved static model training and inference under simulated streaming. Here is a summary:
1. **Offline Model Training:** The training data from `training-dataset.csv` is first aggregated into 5-minute windows, calculating the average fare for each. Instead of using a raw time stamp, we perform feature engineering, creating cyclical features like `hour_of_day` and `minute_of_hour` from the window's start time. A linear regression model is trained on these features and saved.
2. **Real-Time Inference:** The live stream is aggregated using the same 5-minute windowing logic. The same `hour_of_day` and `minute_of_hour` features are created for each window. The pre-trained model is then used to predict the avg_fare for that time window.

Output files from the execution of Task 5 are stored under the directory `models/fare_trend_model_v2`. The Python script `task5.py` contains the code employed for Task 5.

Moreover, we include below two screeshots taken while the script `task5.py` was running:

![Output from task 5](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task5-1.png)

![Printed output in task 5](https://github.com/acintron-a/HO-L8-AC/raw/main/assets/task5-2.png)
