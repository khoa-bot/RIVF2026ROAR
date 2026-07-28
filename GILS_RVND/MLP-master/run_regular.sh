#!/bin/bash

echo "--TRP Benchmark--"

make

k=1
for instance in regularInstances/*; do
	echo $instance >> ./benchmark/GILS_regular.txt

	echo "Running $instance"
	echo "Instance $k of 10" 

	for i in {1..10}; do
		./mlp ${instance} | grep --line-buffered 'COST\|TIME\|PROGRESS' >> ./benchmark/GILS_regular.txt
	done
	k=$(($k + 1))
done

echo "-" >> ./benchmark/GILS_regular.txt

# echo "Running bm.py to compute averages..."

# cd benchmark
# python3 bm.py

echo "Finishing up summary..."
# python3 summarycount.py
cd ..

echo "Benchmark completed."
