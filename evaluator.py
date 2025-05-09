# for data handling 
import numpy as np
import pandas as pd

# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# for function types
from typing import Dict, List, Any

# evaluates the model results
# at cluster-level and region-level
class Evaluator:

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.n_clusters = results[0]["actuals"].shape[0] if results else 0
        self.overall_mape = None
        self.mape_by_cluster = None
        self.mape_df = None
        self.regional_metrics = None

    def get_total_splits_per_region(self, r: Dict[str, int]) -> List[int]:

        total_splits = []
        for i in range(r["regions"]):
            if i < r["remainder"]:
                total_splits.append(r["splits_per_region"] + 1)
            else:
                total_splits.append(r["splits_per_region"])

        return total_splits

    def __compute_error(self, res: Dict[str, Any]) -> np.ndarray:
        
        actuals = res["actuals"].numpy()
        predictions = res["predictions"].cpu().numpy()
        
        epsilon = 1e-10
        
        return np.abs(actuals - predictions) / (actuals + epsilon)

    def compute_regional_mape(self, regions: int = 4) -> Dict[str, Any]:

        n_splits = len(self.results)
        splits_per_region = n_splits // regions
        remainder_splits = n_splits % regions
        
        total_splits = self.get_total_splits_per_region(
            r = {"regions": regions, 
                 "remainder": remainder_splits, 
                 "splits_per_region": splits_per_region}
        )
        
        regional_metrics = {"overall": [], "by_cluster": [], 
                            "region_names": []}
        offset = 0
        
        for region_idx in range(regions):

            end = offset + total_splits[region_idx]
            region_results = self.results[offset:end]
            region_errors = np.stack([self.__compute_error(res) for res in region_results])
            
            regional_metrics["overall"].append(np.mean(region_errors) * 100)
            regional_metrics["by_cluster"].append(np.mean(region_errors, 
                                                          axis=(0, 2)) * 100)
            regional_metrics["region_names"].append(f"Region {region_idx+1}")
            offset = end
        
        self.regional_metrics = regional_metrics

        return regional_metrics

    def compute_global_mape(self) -> None:
        
        all_errors = np.stack([self.__compute_error(res) for res in self.results])
        self.overall_mape = np.mean(a = all_errors) * 100
        self.mape_by_cluster = np.mean(a = all_errors, axis = (0, 2)) * 100

    def forecast_error_by_region(self, regions: int = 4) -> None:

        n_splits = len(self.results)
        splits_per_region = n_splits // regions
        remainder_splits = n_splits % regions
        
        total_splits = self.get_total_splits_per_region(
            r = {
                 "regions": regions, 
                 "remainder": remainder_splits, 
                 "splits_per_region": splits_per_region
                }
        )
        
        offset = 0
        forecast_errors = {"region": [], "errors": []}
        
        for i in range(regions):
            end = offset + total_splits[i]
            region_errors = [np.sum(np.abs(res["actuals"].numpy() - res["predictions"].cpu().numpy())) 
                           for res in self.results[offset:end]]
            forecast_errors["region"].extend([f"Region {i+1}"] * total_splits[i])
            forecast_errors["errors"].extend(region_errors)
            offset = end
        
        plt.figure(figsize = (10, 6))
        sns.boxplot(data = pd.DataFrame.from_dict(forecast_errors), 
                    x = "region", y = "errors")
        plt.title("Forecast Errors by Region")
        plt.show()

    def mape_by_forecast(self) -> None:

        mape_forecast = [self.__compute_error(res) * 100 for res in self.results]
        n_forecasts = np.array(mape_forecast).shape[-1]
        
        self.mape_df = pd.DataFrame(
            data=np.array(mape_forecast).reshape(-1, n_forecasts),
            columns=[f"forecast {i+1}" for i in range(n_forecasts)]
        )
        
        plt.figure(figsize = (10, 6))
        sns.boxplot(data = self.mape_df)
        plt.title("MAPE Distribution by Forecast Step")
        plt.ylabel("MAPE (%)")
        plt.show()

    def print_mape_summary(self) -> None:

        clusters = [1, 3, 5, 7, 8]

        if self.overall_mape is None:
            self.compute_global_mape()

        if self.regional_metrics is None:
            self.compute_regional_mape()
            
        print("Global MAPE Metrics: ")
        print(f"Overall MAPE: {self.overall_mape:.2f}%")
        print("\nMAPE by Cluster:")

        for i, mape in enumerate(self.mape_by_cluster):
            print(f"{clusters[i]}: {mape:.2f}%")
        
        print("\nRegional MAPE Metrics")
        for region_name, overall, by_cluster in zip(
            self.regional_metrics["region_names"],
            self.regional_metrics["overall"],
            self.regional_metrics["by_cluster"]
        ):
            print(f"\n{region_name}:")
            print(f"  Overall: {overall:.2f}%")
            for i, mape in enumerate(by_cluster):
                print(f"{clusters[i]}: {mape:.2f}%")
