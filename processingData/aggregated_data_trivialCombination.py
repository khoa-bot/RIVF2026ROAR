import pandas as pd

def aggregate_trp_runs():
    input_file = "trp_results_ROAR_regularDataset_modified_trivialCombination.csv" # Update to _2 if needed
    output_file = "aggregated_trp_results_trivialCombination.csv"
    
    try:
        # Load the CSV
        df = pd.read_csv(input_file)
        
        # Verify the expected columns exist
        expected_columns = ['Dataset', 'FinalCost', 'Time(s)']
        if not all(col in df.columns for col in expected_columns):
            print(f"Warning: Ensure the file contains these exact columns: {expected_columns}")
            return
            
        # --- THE FIX ---
        # Convert columns to numeric. 
        # errors='coerce' will turn any text (like "FinalCost" in the middle of your data) into NaN
        df['FinalCost'] = pd.to_numeric(df['FinalCost'], errors='coerce')
        df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
        df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
        
        # Drop the rows that contained text headers (which are now NaN)
        df.dropna(subset=['FinalCost', 'Time(s)'], inplace=True)
        # ---------------
            
        # Group by the dataset name and calculate the mean
        aggregated_df = df.groupby(['Dataset', 'Nodes', 'Method'], as_index=False).agg({
            'FinalCost': 'mean',
            'Time(s)': 'mean'
        })
        
        # Rename the output columns 
        aggregated_df.rename(columns={
            'FinalCost': 'Average_Cost',
            'Time(s)': 'Average_Runtime'
        }, inplace=True)
        
        # Sort by Nodes to keep the dataset progression logical
        aggregated_df.sort_values(by='Nodes', inplace=True)
        
        # Export the final aggregated data
        aggregated_df.to_csv(output_file, index=False)
        
        print(f"Success! Aggregated data successfully exported to: {output_file}")
        print("\n--- Preview of Aggregated Data ---")
        print(aggregated_df.head())
        
    except FileNotFoundError:
        print(f"Error: Could not locate '{input_file}'. Ensure it is in the same directory as this script.")

if __name__ == "__main__":
    aggregate_trp_runs()