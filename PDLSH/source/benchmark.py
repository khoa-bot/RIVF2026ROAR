import os
import subprocess
import re

# --- Configuration ---
#Remember to compile the C++ code before running this benchmark script. The executable should be in the same directory as this script or provide the correct path to it.
EXECUTABLE = "time_version.exe" if os.name == 'nt' else "./time_version" #make sure to adjust this if your executable has a different name or path
DATASET_DIR = "../TSPLIB"  # Points to the TSPLIB folder outside the current directory
LOG_FILE = "PDLSH_largeDataset.txt" #change this to "PDLSH_smallDataset.txt" if you want to run the benchmark on smaller datasets

# Constant: Number of times to run each dataset
NUM_EXECUTIONS = 10

# List of tuples: (dataset_filename, node_count)
#If you want to run the benchmark on smaller datasets, uncomment the following section and comment out the large datasets section below.
#Regular Datasets
# DATASETS = [
#     ("berlin52.tsp", 52),
#     ("st70.tsp", 70),
#     ("kroA100.tsp", 100),
#     ("pr107.tsp", 107),
#     ("ch150.tsp", 150),
#     ("d198.tsp", 198),
#     ("ts225.tsp", 225),
#     ("a280.tsp", 280),
#     ("pcb442.tsp", 442),
#     ("att532.tsp", 532),
#     ("rat783.tsp", 783),
#     ("dsj1000.tsp", 1000),
#     ("u1432.tsp", 1432),
#     ("d1655.tsp", 1655),
#     ("pr2392.tsp", 2392),
#     ("pcb3038.tsp", 3038),
#     ("fnl4461.tsp", 4461),
#     ("rl5934.tsp", 5934),
#     ("pla7397.tsp", 7397),
#     ("rl11849.tsp", 11849),
# ]
#large datasets
DATASETS = [
    ("usa13509.tsp", 13509),
    ("fma21553.tsp", 21553),
    ("lsb22777.tsp", 22777),
    ("bbz25234.tsp", 25234),
    ("boa28924.tsp", 28924),
    ("pla33810.tsp", 33810),
]
def get_search_timeout(node_count):
    """
    Returns 100s for < 1000 nodes.
    Grows linearly from 100s (at 1000 nodes) to 1000s (at 100,000 nodes).
    Slope (m) = (1000 - 100) / (100000 - 1000) = 900 / 99000 = 1/110
    """
    if node_count < 1000:
        return 100.0
    
    timeout = 100.0 + ((node_count - 1000) / 110.0)
    return round(timeout, 2)

def run_benchmark():
    # Initialize the log file
    with open(LOG_FILE, "a") as f:
        f.write("=== PDLSH Benchmark Log ===\n\n")

    print(f"Found {len(DATASETS)} datasets. Starting benchmark...\n")

    for filename, node_count in DATASETS:
        dataset_path = os.path.join(DATASET_DIR, filename)
        timeout = get_search_timeout(node_count)
        
        # Verify dataset exists before starting the execution loop
        if not os.path.exists(dataset_path):
            print(f"[WARNING] Dataset not found at {dataset_path}. Skipping.")
            continue

        for run in range(1, NUM_EXECUTIONS + 1):
            print(f"Running {filename} | Nodes: {node_count} | Run: {run}/{NUM_EXECUTIONS} | Timeout: {timeout}s...", end="", flush=True)
            
            # Construct the command array
            command = [EXECUTABLE, dataset_path, str(timeout)]
            
            try:
                # Execute the CUDA program and capture stdout
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                output = process.stdout
                
                # 1. Extract the Best Cost
                cost_match = re.search(r'Absolute Best Final Cost:\s*(\d+)', output)
                best_cost = cost_match.group(1) if cost_match else "ERROR_PARSING_COST"
                
                # 2. Extract and format the Best Route
                # re.DOTALL ensures the regex captures data across multiple lines
                route_match = re.search(r'=== BEST ROUTE FOUND ===\n(.*?)\n========================', output, re.DOTALL)
                if route_match:
                    raw_route = route_match.group(1)
                    # Replace commas with spaces, and split/join to remove trailing whitespace/newlines
                    clean_route = " ".join(raw_route.replace(',', ' ').split())
                else:
                    clean_route = "ERROR_PARSING_ROUTE"
                # 2b. Extract the actual runtime printed by the CUDA program
                time_match = re.search(r'Time Elapsed:\s*([\d.]+)\s*seconds', output)
                actual_runtime = time_match.group(1) if time_match else "ERROR_PARSING_TIME"
                # 3. Log the results
                with open(LOG_FILE, "a") as f:
                    f.write(f"Dataset: {filename} | Execution: {run} | Nodes: {node_count} | Timeout: {timeout}s\n")
                    f.write(f"Best Cost: {best_cost}\n")
                    f.write(f"Best Route: {clean_route}\n")
                    f.write(f"Actual Runtime: {actual_runtime}\n")
                    f.write("-" * 60 + "\n")
                
                print(f" Done! Cost: {best_cost}")

            except subprocess.CalledProcessError as e:
                print(" FAILED!")
                with open(LOG_FILE, "a") as f:
                    f.write(f"Dataset: {filename} | Execution: {run}\n")
                    f.write(f"ERROR: C++ Process crashed or returned non-zero exit code.\n")
                    f.write("-" * 60 + "\n")

    print(f"\nAll benchmark runs complete. Results successfully saved to {LOG_FILE}.")

if __name__ == "__main__":
    run_benchmark()