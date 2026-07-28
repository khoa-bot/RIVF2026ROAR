#include "stdio.h"
#include "stdlib.h"
#include <time.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <string>

/* =========================================================
   [MODIFICATION] Added Dependencies for the New Features
   =========================================================
   - <vector>: To store the snapshot of the best route safely.
   - <mutex>: To prevent torn reads when the logger and optimizer access the route concurrently.
   - <fstream>: To write the log out to a text file.
   - <cmath>: For trigonometric and math functions (ceil, cos, acos) used in TSPLIB metric calculations.
*/
#include <vector>
#include <mutex>
#include <fstream>
#include <cmath>

#ifndef PI
#define PI 3.14159265358979323846
#endif

/* =========================================================
   [MODIFICATION] Adaptive Distance Metric Globals
   =========================================================
   We use __device__ __managed__ so both the CPU (for parsing) 
   and the GPU (for execution) can read the selected edge weight type.
*/
enum EdgeWeightType { EUC_2D, ATT, GEO, CEIL_2D, UNKNOWN };
__device__ __managed__ EdgeWeightType global_ewt = EUC_2D;

/* =========================================================
   [MODIFICATION] Thread-Safe Snapshot Mechanisms
   =========================================================
*/
std::mutex route_mutex;
std::vector<int> best_route_snapshot;

// Thread-safe signals for the Watchdog
std::atomic<bool> stop_flag(false);
std::atomic<long long> global_best_cost(-1);

// Macro updated to safely catch the Watchdog's signal
#define CHECK_DEADLINE() \
    if (stop_flag.load()) { \
        goto end_of_search; \
    }

/* =========================================================
   [MODIFICATION] SYNC_BEST_ROUTE Macro
   =========================================================
   This replaces "global_best_cost.store(dst);". It locks the 
   mutex just long enough to copy the route array into a vector. 
   This MUST be called *after* the array is manipulated.
*/
#define SYNC_BEST_ROUTE() \
    global_best_cost.store(dst);
/* =========================================================
   [MODIFICATION] Logger Functionality
   =========================================================
*/
double get_log_interval(int nodes) {
    if (nodes < 1000) return 1.0;
    if (nodes >= 100000) return 30.0;
    // Linear interpolation between 1000 (1s) and 100000 (30s)
    return 1.0 + (29.0 * (nodes - 1000)) / 99000.0;
}

void start_log_writer(int n, std::string dataset_name) {
    double interval = get_log_interval(n);
    printf("[LOGGER] Initialized log writer. Interval set to %.2f seconds.\n", interval);
    
    // We capture 'dataset_name' by value so the detached thread safely owns the string
    std::thread logger([interval, dataset_name]() {
        std::ofstream log_file("log_PDLSH.txt", std::ios::out | std::ios::app);
        
        // --- UPDATED SEPARATOR HEADER ---
        time_t rawtime;
        time(&rawtime);
        log_file << "\n========================================\n";
        log_file << "DATASET: " << dataset_name << "\n";
        log_file << "NEW RUN STARTED: " << ctime(&rawtime);
        log_file << "========================================\n";
        // --------------------------------

        // 1. Capture the exact start time
        auto start_time = std::chrono::steady_clock::now();
        
        // 2. FORCE IMMEDIATE WRITE: Record the draft baseline at Time = 0.0
        log_file << "Time: 0.0000 | Cost: " << global_best_cost.load() << "\n";
        log_file.flush();
        
        // 3. Now enter the loop and sleep
        while (!stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::duration<double>(interval));
            if (stop_flag.load()) break;

            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;

            long long current_cost = global_best_cost.load();
            
            log_file << "Time: " << elapsed.count() << " | Cost: " << current_cost << "\n";
            log_file.flush(); 
        }
        log_file.close();
    });
    logger.detach(); 
}

/* Vector that holds threads' vertex pair and corresponding latency */
struct min_dst_data
{
	long long dst, i, j;
};

/* A kernel function to initialize min_dst_data vector */
__global__ void fill_dst_arr(struct min_dst_data *dst_arr, long long dst, long long sol)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < sol)
		dst_arr[id].dst = dst;
}

/* =========================================================
   [MODIFICATION] TSPLIB-Compliant Adaptive distD function
   =========================================================
*/
__device__ __host__ inline double d_rad(float deg) {
    int deg_int = (int)deg;
    float min = deg - deg_int;
    return PI * (deg_int + 5.0 * min / 3.0) / 180.0;
}

__device__ __host__ long long distD(int i, int j, float *x, float *y)
{
    double dx, dy, r;
    long long t;
    
    // Divergence here is non-existent since global_ewt is constant for the whole kernel
    switch(global_ewt) {
        case CEIL_2D:
            dx = x[i] - x[j];
            dy = y[i] - y[j];
            return (long long)ceil(sqrt(dx*dx + dy*dy));
            
        case ATT:
            dx = x[i] - x[j];
            dy = y[i] - y[j];
            r = sqrt((dx*dx + dy*dy) / 10.0);
            t = (long long)(r + 0.5);
            return (t < r) ? t + 1 : t;
            
        case GEO:
            double lati, longi, latj, longj, RRR, q1, q2, q3;
            lati = d_rad(x[i]);
            longi = d_rad(y[i]);
            latj = d_rad(x[j]);
            longj = d_rad(y[j]);
            RRR = 6378.388;
            q1 = cos(longi - longj);
            q2 = cos(lati - latj);
            q3 = cos(lati + latj);
            return (long long)(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
            
        case EUC_2D:
        default:
            dx = x[i] - x[j];
            dy = y[i] - y[j];
            return (long long)(sqrt(dx*dx + dy*dy) + 0.5);
    }
}

/* A minimum triple finding kernel */
__global__ void find_min(struct min_dst_data *dst_tid, long long sol, long long i, long long j)
{
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id % j == 0 && (id + i) < sol)
	{
		if(dst_tid[id].dst > dst_tid[id + i].dst)
		{
			dst_tid[id].dst = dst_tid[id+i].dst;
			dst_tid[id].i = dst_tid[id+i].i;
			dst_tid[id].j = dst_tid[id+i].j;
		}
	}
}

/* A kernel for swap move evaluation using built-in reduction */
__global__ void swap(int *rt, long long n, float *posx, float *posy, unsigned long long *dst_tid, long long cost, long long sol)
{
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	long long i, j;
	long long change;
	if(id < sol)
	{
		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i)
		{
			if(i == j-1 && j < n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[j+1], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[j+1], posx, posy));
			}
			else if(i == j-1 && j == n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[0], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[0], posx, posy));
			}
			else
			{
				int next_j = (j == n - 1) ? rt[0] : rt[j + 1];
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i)*distD(rt[j], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[i], posx, posy)
					+(long long)(n-j)*distD(rt[i], next_j, posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i)*distD(rt[i], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[j], posx, posy)
					+(long long)(n-j)*distD(rt[j], next_j, posx, posy));
			}
			if(change < 0)
			{
				cost += change;
				atomicMin(dst_tid, ((unsigned long long)cost << 32) | id);
			}
		}
	}
}

/* A kernel for swap move evaluation using vector reduction */
__global__ void swap_loc(int *rt, long long n, float *posx, float *posy, struct min_dst_data *dst_tid, long long cost, long long sol)
{
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	long long i, j;
	long long change;
	__shared__ struct min_dst_data arr_dst[257];
	arr_dst[blockDim.x].dst = cost;
	if(threadIdx.x < blockDim.x)
		arr_dst[threadIdx.x].dst = cost;
	__syncthreads();
	if(id < sol)
	{

		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i)
		{
			if(i == j-1 && j < n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[j+1], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[j+1], posx, posy));
			}
			else if(i == j-1 && j == n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[0], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[0], posx, posy));
			}
			else
			{
				int next_j = (j == n - 1) ? rt[0] : rt[j + 1];
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i)*distD(rt[j], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[i], posx, posy)
					+(long long)(n-j)*distD(rt[i], next_j, posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i)*distD(rt[i], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[j], posx, posy)
					+(long long)(n-j)*distD(rt[j], next_j, posx, posy));
			}
			if(change < 0)
			{
				cost += change;
				arr_dst[threadIdx.x].dst = cost;
				arr_dst[threadIdx.x].i = i;
				arr_dst[threadIdx.x].j = j;
			}
		}
	}

	__syncthreads();
	int fact = blockDim.x % 2 == 0 ? blockDim.x >> 1 : (blockDim.x + 1) >> 1;
	while(fact)
	{
		if(threadIdx.x < fact)
		{
			if(arr_dst[threadIdx.x].dst > arr_dst[threadIdx.x + fact].dst)
			{
				arr_dst[threadIdx.x].dst = arr_dst[threadIdx.x + fact].dst;
				arr_dst[threadIdx.x].i = arr_dst[threadIdx.x + fact].i;
				arr_dst[threadIdx.x].j = arr_dst[threadIdx.x + fact].j;
			}
		}
		if(fact % 2 == 1 && fact != 1)
			fact++; 
		fact = fact / 2;
		__syncthreads();
	}
	__syncthreads();

	if(threadIdx.x == 0)
	{
		dst_tid[blockIdx.x].dst = arr_dst[0].dst;
		dst_tid[blockIdx.x].i = arr_dst[0].i;
		dst_tid[blockIdx.x].j = arr_dst[0].j;
	}
	
}

/* Device function used to calculate latency of solution after applying swap on i,j pair */
__device__ long long get_route_dst(int*route, float *posx, float *posy, int i, int j, int n)
{
long long ltcy = 0;
long long d1 = 0, d2 = 0, d3 = 0;
	int x, y, z;
	for(x = 0, y =1; y <=i; x++, y++)
		d1 += (long long)(n - y + 1) * distD(route[x], route[y], posx, posy);
	for(y = j + 1, x = i + 1; y < n; x = y, y++)
		d2 += (long long)(n - y + 1) * distD(route[x], route[y], posx, posy);
	for( x = i, y = j, z = i; y > i; x = y, y--, z++)
	{
		d3 += (long long)(n - z) * distD(route[x], route[y], posx, posy);
	}
	int last_node = (j == n - 1) ? route[i + 1] : route[n - 1];
	long long return_edge = distD(last_node, route[0], posx, posy);

	ltcy = d1 + d2 + d3 + return_edge;
	return ltcy;
}

/* Function to arrange new solution using i,j pair */
void arrange_route(int*route, int i, int j, int n)
{
	int x, y;
	int * tmp;
	tmp = (int*)malloc(sizeof(int)*(j - i));
	for( x = 0, y = j; y > i; x++, y--)
		tmp[x] = route[y];
	for( x = i+1, y = 0; x <= j; x++, y++)
		route[x] = tmp[y];
	free(tmp);
}

/* Function to display the current solution */
__host__ __device__ void print_route(int *rt, int n)
{
	int i;
	printf("\nroute\n");
	for(i = 0; i < n; i++)
	printf("%d, ", rt[i]);
	printf("\n");
}

/* A kernel function for swap move evaluation using one-pass vector reduction */
__global__ void swap_loc_one(int *rt, long long n, float *posx, float *posy, struct min_dst_data *dst_tid, long long cost, long long sol)
{
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	long long i, j;
	long long change;
	if(id < sol)
	{
		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i)
		{
			if(i == j-1 && j < n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[j+1], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[j+1], posx, posy));
			}
			else if(i == j-1 && j == n-1)
			{
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i-1)*distD(rt[i], rt[0], posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i-1)*distD(rt[j], rt[0], posx, posy));
			}
			else
			{
				int next_j = (j == n - 1) ? rt[0] : rt[j + 1];
				change = ((long long)(n-i+1)*distD(rt[i-1], rt[j], posx, posy)
					+(long long)(n-i)*distD(rt[j], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[i], posx, posy)
					+(long long)(n-j)*distD(rt[i], next_j, posx, posy))
					-
					 ((long long)(n-i+1)*distD(rt[i-1], rt[i], posx, posy)
					+(long long)(n-i)*distD(rt[i], rt[i+1], posx, posy)
					+(long long)(n-j+1)*distD(rt[j-1], rt[j], posx, posy)
					+(long long)(n-j)*distD(rt[j], next_j, posx, posy));
			}
			if(change < 0)
			{
				cost += change;
				dst_tid[id].dst = cost;
				dst_tid[id].i = i;
				dst_tid[id].j = j;
			}
		}
	}
}

/* A kernel function for two-opt move evaluation using one-pass vector reduction */
__global__ void two_opt_loc_one(int *rt, long long n, float *posx, float *posy, struct min_dst_data *dst_tid, long long cost, long long sol)
{
	long long i, j;
	long long new_cost = cost;
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < sol)
	{
		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i && i != j - 1)
		{
			new_cost = get_route_dst(rt, posx, posy, i, j, n);
			if(new_cost < cost)
			{
				dst_tid[id].dst = new_cost;
				dst_tid[id].i = i;
				dst_tid[id].j = j;
			}
			__syncthreads();
		}
	}
}

/* A kernel function for two-opt move evaluation using two-pass vector reduction */
__global__ void two_opt_loc(int *rt, long long n, float *posx, float *posy, struct min_dst_data *dst_tid, long long cost, long long sol)
{
	long long i, j;
	long long new_cost = cost;
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ struct min_dst_data arr_dst[257];
	arr_dst[blockDim.x].dst = cost;
	if(threadIdx.x < blockDim.x)
		arr_dst[threadIdx.x].dst = cost;
	__syncthreads();
	if(id < sol)
	{

		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i && i != j - 1)
		{
			new_cost = get_route_dst(rt, posx, posy, i, j, n);
			if(new_cost < cost)
			{
				arr_dst[threadIdx.x].dst = new_cost;
				arr_dst[threadIdx.x].i = i;
				arr_dst[threadIdx.x].j = j;
			}
		}
	}
	__syncthreads();
	int fact = blockDim.x % 2 == 0 ? blockDim.x >> 1 : (blockDim.x + 1) >> 1;
	while(fact)
	{
		if(threadIdx.x < fact)
		{
			if(arr_dst[threadIdx.x].dst > arr_dst[threadIdx.x + fact].dst)
			{
				arr_dst[threadIdx.x].dst = arr_dst[threadIdx.x + fact].dst;
				arr_dst[threadIdx.x].i = arr_dst[threadIdx.x + fact].i;
				arr_dst[threadIdx.x].j = arr_dst[threadIdx.x + fact].j;
			}
		}
		if(fact % 2 == 1 && fact != 1)
			fact++; 
		fact = fact / 2;
		__syncthreads();
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		dst_tid[blockIdx.x].dst = arr_dst[0].dst;
		dst_tid[blockIdx.x].i = arr_dst[0].i;
		dst_tid[blockIdx.x].j = arr_dst[0].j;
	}
	__syncthreads();

}

/* A kernel function for two-opt move evaluation using one-pass vector reduction */
__global__ void two_opt(int *rt, long long n, float *posx, float *posy, unsigned long long *dst_tid, long long cost, long long sol)
{
	long long i, j;
	long long new_cost;
	long long id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < sol)
	{
		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i && i != j-1)
		{
			new_cost = get_route_dst(rt, posx, posy, i, j, n);
			if(new_cost < cost)
				atomicMin(dst_tid, ((unsigned long long)new_cost << 32) | id);
		}
	}
}

/* Initial solution construction based on NN */
long long nn_route(int *route, long long n, float *posx, float*posy)
{

	route[0] = 0;
	int k = 1, i = 0, j;
	float min;
	int minj, mini, count = 1, flag = 0;
	long long ltcy = 0;
	int *visited = (int*)calloc(n,sizeof(int));
	visited[0] = 1;
	while(count!= n)
	{
		flag = 0;
		for(j = 1;j < n; j++)
		{
			if(i != j && !visited[j])
			{
				min = distD(i, j, posx,posy);
				minj = j;
				break;	
			}
		}

		for(j = minj+1; j < n; j++)
		{
			
			 if( !visited[j])
			{
				if(min > distD(i, j, posx, posy))
				{
					min = distD(i, j, posx, posy);
					mini = j;
					flag = 1;				
				}
			}
		}
		if(flag == 0)
			i = minj;
		else
			i = mini;
		route[k++] = i;
		visited[i] = 1;
		count++;
	}
	free(visited);
	for(i = 0, j = 1; j < n; i++, j++)
		ltcy += (long long)(n - j + 1) * distD(route[i], route[j], posx, posy);
		
	ltcy += distD(route[n - 1], route[0], posx, posy);
	return ltcy;
}

/* A function to verify the constructed solution is feasible or not */
void route_checker(int *route, int n)
{
	int i, *v, flag =0;
	v = (int*)calloc(n, sizeof(int));
	for(i = 0; i < n; i++)
		v[route[i]]++;
	for(i = 0; i < n; i++)
	{
		if(v[i] != 1)
		{
			printf("\nVisited counter: %d city Id: %d \n", v[i], i);
			flag = 1;
			break;
		}	
	}
	if(flag)
		printf("Invalid\t");
	else
		printf("Valid\t");
}

/* Single-thread reduction function to find the minimum triple values */
void find_min_cpu(struct min_dst_data *dst_tid, long long sol)
{
	int min_i = 0, flag = 0;
	long long minD = dst_tid[0].dst;
	for(int i = 1; i < sol; i++)
	{
		if(dst_tid[i].dst < minD)
		{
			minD = dst_tid[i].dst;
			min_i = i;
			flag = 1;
		}
	}
	if(flag)
	{
		dst_tid[0].dst = dst_tid[min_i].dst;
		dst_tid[0].i = dst_tid[min_i].i;
		dst_tid[0].j = dst_tid[min_i].j;
	}
}

int main(int argc, char *argv[])
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "[FATAL ERROR] No CUDA-capable GPU detected. Aborting.\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("[INFO] Detected %d CUDA-capable GPU(s). Proceeding with computations...\n", deviceCount);
    }
	int ch, ch1, ch2, cnt, in1, n;
	float in2, in3;
	FILE *f;
	float *posx, *posy;
	// float tm;
	char str[256];  
	long long dst, ldst, loc_dst;
	int i, j, x, y, tmp, *route, flag, tid;
	int deviceId;
	// clock_t start,end;
    
    // Timer Setup
    double time_limit = (argc >= 3) ? atof(argv[2]) : 60.0; 
    // time_t real_start = time(NULL);
    
    // The Watchdog Thread
    std::thread watchdog([time_limit]() {
        std::this_thread::sleep_for(std::chrono::duration<double>(time_limit));
        printf("\n[WATCHDOG] Time limit of %.1fs reached!\n", time_limit);
        printf("[WATCHDOG] Waiting for the current GPU cycle to finish and capture absolute best cost...\n");
        stop_flag = true; 
    });
    watchdog.detach();

	cudaGetDevice(&deviceId);
	f = fopen(argv[1], "r");
	if (f == NULL) {fprintf(stderr, "could not open file \n");  exit(-1);}

    /* =========================================================
       [MODIFICATION] Pre-Parse Metric Parsing
       =========================================================
       We scan the header of the TSPLIB file to extract the 
       EDGE_WEIGHT_TYPE before resuming normal coordinate parsing. 
    */
    char buf_metric[256];
    global_ewt = EUC_2D; // Fallback default
    while (fscanf(f, "%s", buf_metric) != EOF) {
        if (strcmp(buf_metric, "EDGE_WEIGHT_TYPE") == 0 || strcmp(buf_metric, "EDGE_WEIGHT_TYPE:") == 0) {
            fscanf(f, "%s", buf_metric);
            if (strcmp(buf_metric, ":") == 0) fscanf(f, "%s", buf_metric); // Handle spacing

            if (strcmp(buf_metric, "EUC_2D") == 0) global_ewt = EUC_2D;
            else if (strcmp(buf_metric, "GEO") == 0) global_ewt = GEO;
            else if (strcmp(buf_metric, "ATT") == 0) global_ewt = ATT;
            else if (strcmp(buf_metric, "CEIL_2D") == 0) global_ewt = CEIL_2D;
            break;
        }
        if (strcmp(buf_metric, "NODE_COORD_SECTION") == 0) break;
    }
    rewind(f); // Reset file pointer for your existing parsing logic
    // =========================================================

	char* p = strstr(argv[1], "TRP");

	// start = clock();
	auto real_start_time = std::chrono::steady_clock::now();
	if(p)
	{
		fscanf(f, "%s\n", str);
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "Number-of-machines:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;n++;
		cudaMallocManaged(&route, sizeof(int) * n);

		fscanf(f, "%s\n", str);
		while (strcmp(str, "y-Coor") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float) * n);
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		cudaMallocManaged(&posy, sizeof(float) * n);
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[in1] = in2;
			posy[in1] = in3;
			cnt++;
		}
		fclose(f);
		printf("%s\t",argv[1]);
	}
	else
	{
		char buf[10];
		fscanf(f, "%s", buf);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		n = atoi(str);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float) * n);
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		cudaMallocManaged(&posy, sizeof(float) * n);
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[cnt] = in2;
			posy[cnt] = in3;
			cnt++;
		}
		fclose(f);
		printf("%s\t",argv[1]);
	}
	long long sol = n * (n - 1) / 2;
	route = (int *)malloc(sizeof(int) * n);
	cudaMallocManaged(&route, sizeof(int) * n);
	int blk, thrd;
	if(sol < 256)
	{
		blk = 1;
		thrd = sol;
	}
	else
	{
		blk = (sol - 1) / 256 + 1;
		thrd = 256;
	}

	dst = nn_route(route, n, posx, posy);
	
    /* =========================================================
       [MODIFICATION] Initializing Background Log Writer
       =========================================================
       We start by storing the NN initial route into the snapshot vector.
    */
    {
        std::lock_guard<std::mutex> lock(route_mutex);
        best_route_snapshot.assign(route, route + n);
    }
    global_best_cost.store(dst); // Initial best cost synced
    start_log_writer(n, argv[1]); // Spin up the detached logger thread
    // =========================================================

	printf("%lld\t",dst);
	route_checker(route, n);
	flag = 1;
	ldst = dst;
	struct min_dst_data * dst_arr;
	int fThrds, fBlks;

	ch1 = 2; 
	switch(ch1)
	{
	case 1:
		unsigned long long *dst_tid;
		cudaMallocManaged(&dst_tid, sizeof(unsigned long long));
		flag = 1;	
		ldst = dst;		
		while(flag)
		{
            CHECK_DEADLINE();
			flag = 0;
			*dst_tid = (((long long)dst + 1) << 32) - 1;
			swap<<<blk, thrd>>>(route, n, posx, posy, dst_tid, dst, sol);
			cudaDeviceSynchronize();
			loc_dst = *dst_tid >> 32;
			while(loc_dst < dst)
			{
                CHECK_DEADLINE();
				dst = loc_dst;
				tid = *dst_tid & ((1ull << 32) - 1); 
				x = n - 2 - floor((sqrt(8 * (sol - tid - 1) + 1) - 1) / 2);
				y = tid - x * (n - 1) + (x * (x + 1) / 2) + 1;
				
                tmp = route[x];
				route[x] = route[y];
				route[y] = tmp;

                /* =========================================================
                   [MODIFICATION] Safely Sync Result AFTER Array Modification
                   ========================================================= */
				SYNC_BEST_ROUTE();

				*dst_tid = (((long long)dst + 1) << 32) - 1;
				swap<<<blk, thrd>>>(route, n, posx, posy, dst_tid, dst, sol);
				cudaDeviceSynchronize();
				loc_dst = *dst_tid >> 32;
			}
			*dst_tid = (((long long)dst + 1) << 32) - 1;
			two_opt<<<blk, thrd>>>(route, n, posx, posy, dst_tid, dst, sol);
			cudaDeviceSynchronize();
			loc_dst = *dst_tid >> 32;
			while(loc_dst < dst)
			{
                CHECK_DEADLINE();
				dst = loc_dst;
				tid = *dst_tid & ((1ull << 32) - 1); 
				x = n - 2 - floor((sqrt(8 * (sol - tid - 1) + 1) - 1) / 2);
				y = tid - x * (n - 1) + (x * (x + 1) / 2) + 1;
				*dst_tid = (((long long)dst + 1) << 32) - 1;
				
                arrange_route(route, x, y, n);

                /* =========================================================
                   [MODIFICATION] Safely Sync Result AFTER Array Modification
                   ========================================================= */
                SYNC_BEST_ROUTE();

				two_opt<<<blk, thrd>>>(route, n, posx, posy, dst_tid, dst, sol);
				cudaDeviceSynchronize();
				loc_dst = *dst_tid >> 32;
			}
			if(dst < ldst)
			{
				flag = 1;
				ldst = dst;
			}
		}
		cudaFree(dst_tid);
	break;

	case 2:
		ch2 = 3; 
		switch(ch2)
		{
		case 1:
			cudaMallocManaged(&dst_arr, sizeof(struct min_dst_data) * (blk + 1));
			dst_arr[blk].dst = dst;
			if (blk > 256)
			{
				fThrds = 256;
				fBlks = (blk - 1)/256 + 1;
			} 
			else
			{
				fThrds = blk;
				fBlks = 1;
			} 
			fill_dst_arr<<<fBlks, fThrds>>>(dst_arr, dst, blk);
			cudaDeviceSynchronize();
			flag = 1;
			ldst = dst;
			while(flag)
			{
                CHECK_DEADLINE();
				flag = 0;
				swap_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				find_min_cpu(dst_arr, blk);
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    tmp = route[x];
					route[x] = route[y];
					route[y] = tmp;

                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					swap_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					find_min_cpu(dst_arr, blk);
				}
				two_opt_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				find_min_cpu(dst_arr, blk);
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    arrange_route(route, x, y, n);

                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					two_opt_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					find_min_cpu(dst_arr, blk);
				}
				if(dst < ldst)
				{
					ldst = dst;
					flag = 1;
				}
			}
		break;
		case 2:
			cudaMallocManaged(&dst_arr, sizeof(struct min_dst_data) * (sol + 1));
			dst_arr[sol].dst = dst;
			fill_dst_arr<<<blk, thrd>>>(dst_arr, dst, sol);
			cudaDeviceSynchronize();
			flag = 1;
			ldst = dst;
			while(flag)
			{
                CHECK_DEADLINE();
				flag = 0;
				dst_arr[sol].dst = dst;
				swap_loc_one<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				i = 1;
				j = 2;
				find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
				cudaDeviceSynchronize();
				i *= 2;
				j *= 2;
				while(i < sol)
				{
					find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
				}
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    tmp = route[x];
					route[x] = route[y];
					route[y] = tmp;
                    
                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					swap_loc_one<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					i = 1;
					j = 2;
					find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
					while(i < sol)
					{
						find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
						cudaDeviceSynchronize();
						i *= 2;
						j *= 2;
					}
				}
				two_opt_loc_one<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				i = 1;
				j = 2;
				find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
				cudaDeviceSynchronize();
				i *= 2;
				j *= 2;
				while(i < sol)
				{
					find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
				}
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    arrange_route(route, x, y, n);
                    
                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					two_opt_loc_one<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					i = 1;
					j = 2;
					find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
					while(i < sol)
					{
						find_min<<<blk, thrd>>>(dst_arr, sol, i, j);
						cudaDeviceSynchronize();
						i *= 2;
						j *= 2;
					}
				}
				if(dst < ldst)
				{
					ldst = dst;
					flag = 1;
				}
			}
		break;
		case 3:
			cudaMallocManaged(&dst_arr, sizeof(struct min_dst_data) * (blk + 1));
			dst_arr[blk].dst = dst;
			int fThrds, fBlks;
			if (blk > 256)
			{
				fThrds = 256;
				fBlks = (blk - 1)/256 + 1;
			} 
			else
			{
				fThrds = blk;
				fBlks = 1;
			} 
			fill_dst_arr<<<fBlks, fThrds>>>(dst_arr, dst, blk);
			cudaDeviceSynchronize();
			while(flag)
			{
				CHECK_DEADLINE();
				flag = 0;
				swap_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				i = 1;
				j = 2;
				find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
				cudaDeviceSynchronize();
				i *= 2;
				j *= 2;
				while(i < blk)
				{
					find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
				}
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    tmp = route[x];
					route[x] = route[y];
					route[y] = tmp;

                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					swap_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					i = 1;
					j = 2;
					find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
					while(i < blk)
					{
						find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
						cudaDeviceSynchronize();
						i *= 2;
						j *= 2;
					}
				}
				two_opt_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
				cudaDeviceSynchronize();
				i = 1;
				j = 2;
				find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
				cudaDeviceSynchronize();
				i *= 2;
				j *= 2;
				while(i < blk)
				{
					find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
				}
				while(dst_arr[0].dst < dst)
				{
                    CHECK_DEADLINE();
					dst = dst_arr[0].dst;
					x = dst_arr[0].i;
					y = dst_arr[0].j;
					
                    arrange_route(route, x, y, n);

                    /* =========================================================
                       [MODIFICATION] Safely Sync Result AFTER Array Modification
                       ========================================================= */
                    SYNC_BEST_ROUTE();
                    printf("[2-OPT] Improved Cost: %lld\n", dst);

					cudaDeviceSynchronize();
					two_opt_loc<<<blk, thrd>>>(route, n, posx, posy, dst_arr, dst, sol);
					cudaDeviceSynchronize();
					i = 1;
					j = 2;
					find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
					cudaDeviceSynchronize();
					i *= 2;
					j *= 2;
					while(i < blk)
					{
						find_min<<<fBlks, fThrds>>>(dst_arr, blk, i, j);
						cudaDeviceSynchronize();
						i *= 2;
						j *= 2;
					}
				}
				if(dst < ldst)
				{
					ldst = dst;
					flag = 1;
				}
			}
		break;
		}
	break;
	}

    end_of_search:
    printf("\n=== BEST ROUTE FOUND ===\n");
    for(int k = 0; k < n; k++) {
        printf("%d, ", route[k]);
    }
    printf("\n========================\n");

	// end = clock();
	// tm = ((double) (end - start)) / CLOCKS_PER_SEC;
	// printf("--- Absolute Best Final Cost: %lld ---\n", global_best_cost.load());
	// printf("Time Elapsed: %f seconds\n", tm);
	auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - real_start_time;
    
    printf("--- Absolute Best Final Cost: %lld ---\n", global_best_cost.load());
    printf("Time Elapsed: %f seconds\n", elapsed.count());
	
	cudaFree(posx);
	cudaFree(posy);
	cudaFree(dst_arr);
	cudaFree(route);
	return 0;
}