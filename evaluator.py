import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:

    def __init__(self, results: dict[str, any]):
        
        self.results = results

    def get_total_splits_per_region(self, r: dict[str, int]):

        total_splits_across_regions = []

        for i in range(r["regions"]):

            if i < r["remainder"]:

                total_splits_across_regions.append(r["splits_per_region"] + 1)

            else:

                total_splits_across_regions.append(r["splits_per_region"])
        
        return total_splits_across_regions
    
    def compute_forecast_error(self, window_res: dict[str, any]):

        return np.sum(np.abs(window_res["actuals"] - window_res["predictions"]))

    def compute_mape(self, window_res: dict[str, any]):

        error = window_res["actuals"] - window_res["predictions"]
        return np.abs((error)/(window_res["actuals"])) * 100

    def forecast_error_by_region(self, regions: int = 4):

        n_splits = len(self.results)

        splits_per_region = n_splits // regions
        remainder_splits = n_splits % regions

        total_splits_across_regions = self.get_total_splits_per_region(
                                                r = {
                                                    "regions": regions,
                                                    "remainder": remainder_splits,
                                                    "splits_per_region": splits_per_region
                                                }
                                            )
        
        offset = 0
        forecast_errors = {
            "region": [],
            "errors": []
        }
        for i in range(regions):
            
            end = offset + total_splits_across_regions[i]
            forecast_errors_by_region = list(map(self.compute_forecast_error, 
                                                 self.results[offset:end]))
            forecast_errors["region"].extend([f"Region {i + 1}"] * (total_splits_across_regions[i]))
            forecast_errors["errors"].extend(forecast_errors_by_region)
            offset = end
        
        sns.boxplot(
            data = pd.DataFrame.from_dict(forecast_errors),
            x = "region",
            y = "errors"
        )

    def mape_by_forecast(self):

        mape_forecast = list(map(self.compute_mape, self.results))
        n_forecasts = np.array(mape_forecast).shape[-1]

        #print(mape_forecast[0])

        self.mape_df = pd.DataFrame(
                        data = np.array(mape_forecast).reshape(-1, n_forecasts),
                        columns = [f"forecast {i + 1}" for i in range(n_forecasts)] 
                  )
        
        sns.boxplot(
            data = self.mape_df,
        )