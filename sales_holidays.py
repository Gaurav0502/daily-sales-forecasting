# for data handling
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# constants
EVENT_COLORS = {
    "easter_season": "pink",
    "back_to_school_sale": "green",
    "black_friday_sale": "crimson",
    "boxing_day_sale": "cyan",
    "school_holidays": "gold"
}

# gives patch aesthetic to the event
def get_patch(event: str, color: str):
    return mpatches.Patch(color = color, 
                          alpha = 0.3, 
                          label = event.replace("_", " ").title())

# returns all labels and handles for the legend
def get_all_labels_and_handles(axis: any, event_colors: dict[str, str]):

    patches = [get_patch(event, color) for event, color in event_colors.items()]
    
    line_handles = [
        axis.lines[0],
        axis.lines[1] 
    ]

    line_labels = ["Total Sales", "Days to Christmas"]

    all_handles = patches + line_handles
    all_labels = [p.get_label() for p in patches] + line_labels

    return all_handles, all_labels

# plots time series by overlaying events
def plot_ts_wt_events(data: pd.DataFrame, 
                      event_ranges: list, 
                      event_colors: dict[str, str]) -> None:

    clusters = sorted(data['cluster_id'].unique())
    n_clusters = len(clusters)
    
    fig, axes = plt.subplots(nrows = n_clusters, 
                             ncols = 1, 
                             figsize = (12, 3 * n_clusters), 
                             sharex=True)
    
    all_handles = None
    all_labels = None
    
    for i, cluster in enumerate(clusters):
        cluster_data = data[data['cluster_id'] == cluster]
        axes[i].plot(cluster_data['date'], cluster_data['TotalSales'], label=f'Cluster {cluster}')
        axes[i].plot(cluster_data['date'], cluster_data['days_to_christmas'], label=f'Days to christmas')
    
        for event, ranges in event_ranges.items():
            color = event_colors.get(event, "lightcoral")
            for start_str, end_str in ranges:
                start = pd.to_datetime(start_str)
                end = pd.to_datetime(end_str)
                if start >= data["date"].min():
                   axes[i].axvspan(start, end, color=color, alpha=0.3)
    
        if i == 0:
            all_handles, all_labels = get_all_labels_and_handles(axes[i], EVENT_COLORS)
    
        axes[i].set_title(f'{cluster}')
        axes[i].set_ylabel('Total Sales')
        axes[i].grid(True)
        axes[i].legend(handles = all_handles, 
                       labels = all_labels, 
                       loc = 'center left', 
                       bbox_to_anchor = (1.02, 0.5))
        
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()