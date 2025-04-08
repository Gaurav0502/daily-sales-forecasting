import pandas as pd
import numpy as np
import os

DATA = "data"

def get_daily_sales(cluster_id: int) -> pd.DataFrame:

    """
         Aggregates the data to resemble daily sales across all the products in the cluster.

         Args:
            cluster_id: the ID of the cluster (between 0 and 9, both inclusive).

         Returns:
            A dataframe with date and corresponding daily sales.
      """
    
    if cluster_id not in range(10):
        raise ValueError("Cluster ID must be between 0 and 9, both inclusive.")
    
    csv_file = os.path.join(DATA, "clusters", f"cluster_{cluster_id}.csv")

    df = pd.read_csv(csv_file)

    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%m/%d/%Y")
    
    return df[["date", "TotalSales"]].groupby(by = ["date"]).sum(["TotalSales"])