import pandas as pd

def calculate_gaps_for_trp(input_files):
    for file in input_files:
        try:
            # 1. Load the dataset
            df = pd.read_csv(file)
            
            # 2. Define the core cost columns based on the CSV pattern
            roar_cost = 'ROAR_AvgCost'
            pdlsh_cost = 'PDLSH_AvgCost'
            gils_cost = 'GILS-RVND_AvgCost'
            
            # 3. Safely convert to numeric, turning 'OOM' or other text into NaN
            df[roar_cost] = pd.to_numeric(df[roar_cost], errors='coerce')
            df[pdlsh_cost] = pd.to_numeric(df[pdlsh_cost], errors='coerce')
            df[gils_cost] = pd.to_numeric(df[gils_cost], errors='coerce')
            
            # 4. Calculate percentage gaps
            # Positive gap (%) means the competing method costs more than ROAR
            df['Gap_PDLSH_vs_ROAR (%)'] = ((df[pdlsh_cost] - df[roar_cost]) / df[roar_cost]) * 100
            df['Gap_GILS-RVND_vs_ROAR (%)'] = ((df[gils_cost] - df[roar_cost]) / df[roar_cost]) * 100
            
            # 5. Export to a new CSV file
            output_file = file.replace('.csv', '_with_gaps.csv')
            df.to_csv(output_file, index=False)
            
            print(f"Successfully processed '{file}'. Saved output to '{output_file}'.")
            
        except FileNotFoundError:
            print(f"Error: The file '{file}' could not be found.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file}': {e}")

if __name__ == "__main__":
    # Exact verbatim filenames requested
    files_to_process = [
        "trp_summary_regular.csv",
        "trp_summary_large.csv"
    ]
    
    calculate_gaps_for_trp(files_to_process)