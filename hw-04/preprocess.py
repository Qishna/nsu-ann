import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    week = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    data["Day_of_week"] = data["Day_of_week"].map(week)
    data["WeekStatus"] = data["WeekStatus"].map({"Weekday": 0, "Weekend": 1})
    data["Load_Type"] = data["Load_Type"].map(
        {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    )
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    day_len_seconds = 24 * 60 * 60
    year_len_seconds = (365.2425) * day_len_seconds

    data["Time_sin"] = np.sin(
        (data["date"] - data["date"].min()).dt.total_seconds()
        * (2 * np.pi / day_len_seconds)
    )
    data["Date_sin"] = np.sin(
        (data["date"] - data["date"].min()).dt.total_seconds()
        * (2 * np.pi / year_len_seconds)
    )
    data = data.drop(columns=["date"])

    feature_columns = list(data.columns)
    feature_columns.remove("Usage_kWh")
    data[feature_columns] = MinMaxScaler().fit_transform(data[feature_columns])
    return data
