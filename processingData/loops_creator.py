import re
import csv
import statistics
from collections import defaultdict

def parse_logs_and_summarize(file_names, output_csv):
    # Dictionary to store a list of loop counts for each dataset
    dataset_loops = defaultdict(list)
    
    # Regex patterns based on the log structure
    dataset_pattern = re.compile(r"===\s*Processing Dataset:\s*(.*?)\s*\|\s*Method:")
    loops_pattern = re.compile(r"Total LNS Loops Executed:\s*(\d+)")
    
    for file_name in file_names:
        try:
            with open(file_name, 'r') as file:
                current_dataset = None
                
                for line in file:
                    # 1. Look for the dataset name
                    ds_match = dataset_pattern.search(line)
                    if ds_match:
                        current_dataset = ds_match.group(1).strip()
                        continue
                    
                    # 2. Look for the loop count associated with the current dataset
                    loop_match = loops_pattern.search(line)
                    if loop_match and current_dataset:
                        loop_count = int(loop_match.group(1))
                        dataset_loops[current_dataset].append(loop_count)
                        
                        # Reset current_dataset to avoid assigning loops to the wrong dataset 
                        # if the log format unexpectedly breaks
                        current_dataset = None 
                        
        except FileNotFoundError:
            print(f"Warning: The file '{file_name}' was not found. Skipping...")

    # 3. Calculate statistics and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write headers
        csv_writer.writerow(['datasets name', 'average loops', 'min-loop', 'max loop', 'stdev'])
        
        for dataset, loops in dataset_loops.items():
            if not loops:
                continue
            
            # Calculate required metrics
            avg_loops = sum(loops) / len(loops)
            min_loop = min(loops)
            max_loop = max(loops)
            # stdev requires at least 2 data points; default to 0.0 if only 1 run was found
            stdev = statistics.stdev(loops) if len(loops) > 1 else 0.0 
            
            # Write the row, formatting floats to 2 decimal places for cleanliness
            csv_writer.writerow([
                dataset, 
                f"{avg_loops:.2f}", 
                min_loop, 
                max_loop, 
                f"{stdev:.2f}"
            ])
            
    print(f"Successfully processed logs. Summary saved to '{output_csv}'.")

if __name__ == "__main__":
    # Define your input files and the desired output file
    target_files = [
        "log_report_ROAR_extremeDataset_modified.txt", 
        "log_report_ROAR_regularDataset_modified.txt", 
        "log_report_ROAR_largeDataset_modified.txt"
    ]
    output_filename = "roar_loops_summary.csv"
    
    parse_logs_and_summarize(target_files, output_filename)