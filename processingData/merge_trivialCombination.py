import pandas as pd

def merge_trivial_combination():
    summary_file = "trp_summary_regular.csv"
    trivial_file = "aggregated_trp_results_trivialCombination.csv"
    output_file = "trp_summary_regular_updated.csv"
    
    try:
        # Load the two CSV files
        df_summary = pd.read_csv(summary_file)
        df_trivial = pd.read_csv(trivial_file)
        
        # Keep only the necessary columns from the trivial combination file
        # We drop the 'Method' column since the new column names will identify the method
        df_trivial_clean = df_trivial[['Dataset', 'Nodes', 'Average_Cost', 'Average_Runtime']].copy()
        
        # Rename the columns to match the naming convention in the summary file
        df_trivial_clean.rename(columns={
            'Average_Cost': 'ROAR_trivialCombination_AvgCost',
            'Average_Runtime': 'ROAR_trivialCombination_AvgTime(s)'
        }, inplace=True)
        
        # Merge the datasets together based on Dataset and Nodes
        # Using a left merge ensures we keep all rows from the original summary file
        df_merged = pd.merge(df_summary, df_trivial_clean, on=['Dataset', 'Nodes'], how='left')
        
        # Export the updated summary to a new CSV file
        df_merged.to_csv(output_file, index=False)
        
        print(f"Success! The updated summary has been saved to: {output_file}")
        print("\n--- Preview of Updated Summary ---")
        print(df_merged.head())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both 'trp_summary_regular.csv' and 'aggregated_trp_results_trivialCombination.csv' are in the same folder as this script.")

if __name__ == "__main__":
    merge_trivial_combination()