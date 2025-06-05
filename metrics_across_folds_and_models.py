import os
import pandas as pd

# Define your actual path here
base_dir = "/home/rohankshah/state-sets"

# Output file path
output_file = os.path.join(base_dir, "summary_metrics.csv")

# Collect results
summary_data = []

# Loop over all subdirectories starting with 'replogle_'
for folder in os.listdir(base_dir):
    if folder.startswith("replogle_"):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Find all CSV files in the folder ending with '_metrics_avg.csv'
            csv_files = [f for f in os.listdir(folder_path) if f.endswith("_metrics_avg.csv")]
            dfs = []

            for csv_file in csv_files:
                file_path = os.path.join(folder_path, csv_file)
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            # Average across all found files in this directory
            if dfs:
                combined_df = pd.concat(dfs)
                avg_metrics = combined_df.mean()
                avg_metrics["directory"] = folder
                summary_data.append(avg_metrics)

# Create a final DataFrame from the summary data
summary_df = pd.DataFrame(summary_data)
summary_df.set_index("directory", inplace=True)

# Save to CSV
summary_df.to_csv(output_file)
