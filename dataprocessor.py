# for data handling
import pandas as pd
import numpy as np

# for file handling
import os
import json

# for common UK holidays
import holidays

# for operations on dates
from datetime import datetime

# class to add features
# and preprocess the dataset
class DataProcessor:

    def __init__(self, data_dir: str, clusters: list = [1, 5, 7]):
        
        # inputs
        self.DATA_DIR = data_dir
        self.CLUSTERS = clusters
        self.data = None
        self.n = None

        # constants
        self.EVENTS_FP = open("/kaggle/input/events/events.json", 
                              "r")
        self.EVENTS = json.load(self.EVENTS_FP)

        self.uk_holidays = holidays.country_holidays('UK')
    
    # converts data type to category
    def __to_category(self, x: pd.Series) -> pd.Series:
        return x.astype(str).astype("category")
    
    # adds cluster index into the data frame
    def __add_cluster_idx(self, df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
        df["cluster_id"] = f"cluster_{cluster_id}"
        return df
    
    # adds date related features
    def __add_date_features(self):

        relative_date = self.data["date"] - self.data["date"].min()
        self.data["time_idx"] = relative_date.dt.days

        self.data["month"] = self.__to_category(self.data["date"].dt.month)
        self.data["year"] = self.__to_category(self.data["date"].dt.year)
        self.data["day_of_week"] = self.__to_category(self.data["date"].dt.dayofweek)
        self.data["day_of_month"] = self.__to_category(self.data["date"].dt.day)
        self.data["is_weekend"] = (self.data["date"].dt.dayofweek >= 5).astype(int)
    
    # adds one-hot holiday features in-place to the dataset
    def __add_holiday_features(self, feat_name: str, event_date_ranges: dict[str, list[str]]) -> None:

        flag = np.zeros(self.n, dtype = "int64")
        for start_dt, end_dt in event_date_ranges:
            start, end = pd.to_datetime(start_dt), pd.to_datetime(end_dt)
            flag |= self.data['date'].between(start, end).astype(int)

        self.data[feat_name] = flag

    # adds time-related features
    def __add_ts_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by = ["date"])
        
        df['lag_7'] = df['TotalSales'].shift(7)
  
        df['rolling_mean_7'] = df['TotalSales'].rolling(7).mean().ffill()
        return df.dropna()
    
    # removes outliers from the dataset based IQR method
    def __remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        Q1 = df['TotalSales'].quantile(0.25)
        Q3 = df['TotalSales'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['TotalSales'] >= (Q1 - 1.5 * IQR)) & (df['TotalSales'] <= (Q3 + 1.5 * IQR))]

        return df

    # completes the time index by
    # linear interpolation
    def __interpolate_ts(self, df: pd.DataFrame) -> pd.DataFrame:

        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        max_date = df["date"].max()
        idx = pd.date_range(start = min_date, end = max_date, freq = 'D')
        df = df.sort_values(by = ["date"], ascending = True)
        df = df.set_index("date").reindex(idx).interpolate(method = "linear").reset_index()
        df = df.rename(columns={'index': 'date'})

        return df
    
    # computes and returns the total sales
    # of each cluster
    def get_daily_sales(self, cluster_id: int) -> pd.DataFrame:
    
        if cluster_id not in range(10):
            raise ValueError("Cluster ID must be between 0 and 9, both inclusive.")
        
        csv_file = os.path.join(self.DATA_DIR, f"cluster_{cluster_id}.csv")

        data = pd.read_csv(csv_file)

        data["date"] = pd.to_datetime(data["InvoiceDate"]).dt.strftime("%m/%d/%Y")

        data = self.__remove_outliers(data)

        return data[["date", "TotalSales"]].groupby(by = ["date"]).sum(["TotalSales"])
    
    # driver's code
    def process(self):
        
        data = list(map(lambda x: self.__interpolate_ts(self.get_daily_sales(x).reset_index()), 
                         self.CLUSTERS))
        
        data = list(map(self.__add_cluster_idx,
                        data,
                        self.CLUSTERS))

        data = list(map(self.__add_ts_features, data))

        self.data = pd.concat(data).reset_index(drop = True)

        self.data["cluster_id"] = self.data["cluster_id"].astype("category")

        self.data["date"] = pd.to_datetime(self.data["date"])
        self.n = len(self.data)

        self.__add_date_features()

        for feature, ranges in self.EVENTS.items():
            self.__add_holiday_features(feature, ranges)

        self.data['is_holiday'] = self.data['date'].isin(self.uk_holidays).astype(int)

        
        self.data['days_to_christmas'] = self.data['date'].apply(lambda d: (datetime(year = d.year, 
                                                                                     month = 12, 
                                                                                     day = 25) - d).days)
        
        self.data['days_to_christmas'] = self.data['days_to_christmas'].apply(lambda d: 365 + d if d < 0 else d)

        self.data = self.data.sort_values(by = "time_idx").reset_index(drop = True)