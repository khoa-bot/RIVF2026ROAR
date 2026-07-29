#!/bin/bash

# 1. Compile the C++ files
# Using -O3 for maximum performance and -pthread because your code uses <thread>
echo "=== Compiling Programs ==="
g++ -O3 -pthread main_roar.cpp -o run_regular
g++ -O3 -pthread main_ROAR_large.cpp -o run_large
g++ -O3 -pthread main_roar_extreme.cpp -o run_extreme
echo "Compilation successful."
echo "-----------------------------------"

# 2. Define the automated input
# \n acts as the "Enter" key for the default values, and 'y\n' triggers the auto-advance
INPUT_SEQUENCE="\n\n\ny\n"

# 3. Run the executables iteratively
echo "=== Running Regular Dataset Benchmark ==="
printf "$INPUT_SEQUENCE" | ./run_regular

echo ""
echo "=== Running Large Dataset Benchmark ==="
printf "$INPUT_SEQUENCE" | ./run_large

echo ""
echo "=== Running Extreme Dataset Benchmark ==="
printf "$INPUT_SEQUENCE" | ./run_extreme

echo ""
echo "=== All benchmarks completed successfully! ==="