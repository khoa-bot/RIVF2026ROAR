#include"stdio.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include"math.h"
#include <ctype.h>
#include <assert.h>

/* Vector that holds  threads' vertex pair and corresponding latency */
struct min_dst_data
{
	long dst, i, j;
};

/* A kernel function to initialize min_dst_data vector */
__global__ void fill_dst_arr(struct min_dst_data *dst_arr, long dst, long sol)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < sol)
		dst_arr[id].dst = dst;
}

/* Euclidean distance calculation */
__device__ __host__ float distD(int i, int j, float *x, float*y)
{
	float dx = x[i] - x[j];
	float dy = y[i] - y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

/* Initial solution construction based on sequenced order */
float seq_route(int*route,int n, float *posx, float *posy)
{
	int i,j;
	float ltcy = 0;
	for(i = 0; i < n; i++)
		route[i] = i;
	printf("\nInitial solution\n");
	for(i = 0; i < n; i++)
		printf("%d,", route[i]);
	printf("\n");
	for(i = 0, j = 1; j < n; i++, j++)
		ltcy += (n - j) * distD(route[i], route[j], posx, posy);
	return ltcy;
}

/* Initial solution construction using NN */
float nn_route(int *route, long n, float *posx, float*posy)
{

	route[0] = 0;
	int k = 1, i = 0, j;
	float min;
	int minj, mini, count = 1, flag = 0;
	float ltcy = 0;
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
		ltcy += (n - j) * distD(route[i], route[j], posx, posy);
	return ltcy;
}
/* A minimum triple finding kernel */
__global__ void find_min(struct min_dst_data *dst_tid, long sol, long i, long j)
{
	long id = threadIdx.x + blockIdx.x * blockDim.x;
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
/* A kernel for swap move evaluation */
__global__ void swap(int *rt, long n, float *posx, float *posy, struct min_dst_data *dst_tid, long cost, long sol)
{
	long id = threadIdx.x + blockIdx.x * blockDim.x;
	long i, j, long_change;
	float change;
	if(id < sol)
	{
		i = n - 2 - floorf(((int)__dsqrt_rn(8*(sol - id - 1) + 1) - 1) / 2);
		j = id - i * (n - 1) + (i * (i + 1) / 2) + 1;
		if(i)
		{
			if(i == j-1 && j < n-1)
			{
				change = ((n-i)*distD(rt[i-1], rt[j], posx, posy)
					+(n-i-2)*distD(rt[i], rt[j+1], posx, posy))
					-
					 ((n-i)*distD(rt[i-1], rt[i], posx, posy)
					+(n-i-2)*distD(rt[j], rt[j+1], posx, posy));
			}
			else if(i == j-1 && j == n-1)
			{
				change = (n-i)*distD(rt[i-1], rt[j], posx, posy)
					-
					 (n-i)*distD(rt[i-1], rt[i], posx, posy);
			}
			else
			{
				change = ((n-i)*distD(rt[i-1], rt[j], posx, posy)
					+(n-i-1)*distD(rt[j], rt[i+1], posx, posy)
					+(n-j)*distD(rt[j-1], rt[i], posx, posy)
					+(n-j-1)*distD(rt[i], rt[j+1], posx, posy))
					-
					 ((n-i)*distD(rt[i-1], rt[i], posx, posy)
					+(n-i-1)*distD(rt[i], rt[i+1], posx, posy)
					+(n-j)*distD(rt[j-1], rt[j], posx, posy)
					+(n-j-1)*distD(rt[j], rt[j+1], posx, posy));
			}
			long_change = change * 100;
			if(long_change < 0)
			{
				dst_tid[id].dst = cost + long_change;
				dst_tid[id].i = i;
				dst_tid[id].j = j;
			}
		}
	}
}

/* Device function used to calculate latency of solution after applying swap on i,j pair */
__device__ long get_route_dst(int*route, float *posx, float *posy, int i, int j, int n)
{
	long ltcy = 0;
	float d1 = 0, d2 = 0, d3 = 0;
	int x, y, z;
	for(x = 0, y =1; y <=i; x++, y++)
		d1 += (n - y) * distD(route[x], route[y], posx, posy);
	for(y = j + 1, x = i + 1; y < n; x = y, y++)
		d2 += (n - y) * distD(route[x], route[y], posx, posy);
	for( x = i, y = j, z = i; y > i; x = y, y--, z++)
		d3 += (n - z -1) * distD(route[x], route[y], posx, posy);
	ltcy = (d1 + d2 + d3) * 100;
	return ltcy;
}

/* Function to arrange new solution using i,j pair */
void arrange_route(int*route, int i, int j, int n)
{
	int x, y;
	int * tmp;
	tmp = (int*)malloc(sizeof(int)*(j - i));
	for( x = 0, y = j; y > i; x++, y--)
	{
		tmp[x] = route[y];
	}
	for( x = i+1, y = 0; x <= j; x++, y++)
	{
		route[x] = tmp[y];
	}
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

/* A kernel function for two-opt move evaluation */
__global__ void two_opt(int *rt, long n, float *posx, float *posy, struct min_dst_data *dst_tid, long cost, long sol)
{
	long i, j;
	long new_cost;
	long id = threadIdx.x + blockIdx.x * blockDim.x;
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
		}
	}
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
int main(int argc, char *argv[])
{
	int ch, cnt, in1, n;
	float in2, in3;
	FILE *f;
	float *posx, *posy;
	float tm;
	char str[256];  
	float dst;
	long ldst, ld1;
	int i, j, x, y, tmp, *route, flag;
	int deviceId;
	clock_t start,end;
	
	cudaGetDevice(&deviceId);
	f = fopen(argv[1], "r");
	if (f == NULL) {fprintf(stderr, "could not open file \n");  exit(-1);}
	char* p = strstr(argv[1], "TRP");

	start = clock();
	if(p)
	{
		fscanf(f, "%s\n", str);
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "Number-of-machines:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;n++;
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
	long sol = n * (n - 1) / 2;
	route = (int *)malloc(sizeof(int) * n);
	cudaMallocManaged(&route, sizeof(int) * n);
	struct min_dst_data * dst_arr;
	cudaMallocManaged(&dst_arr, sizeof(struct min_dst_data) * (sol + 1));
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
	ldst = dst * 100;
	printf("%f\t", dst);
	flag = 1;
	cnt = 0;
	ld1 = ldst;
	dst_arr[sol].dst = ldst;
	fill_dst_arr<<<blk, thrd>>>(dst_arr, ldst, sol);
	cudaDeviceSynchronize();
	while(flag)
	{
		flag = 0;
		swap<<<blk, thrd>>>(route, n, posx, posy, dst_arr, ldst, sol);
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
		while(dst_arr[0].dst < ldst)
		{
			ldst = dst_arr[0].dst;
			x = dst_arr[0].i;
			y = dst_arr[0].j;
			tmp = route[x];
			route[x] = route[y];
			route[y] = tmp;
			swap<<<blk, thrd>>>(route, n, posx, posy, dst_arr, ldst, sol);
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
		two_opt<<<blk, thrd>>>(route, n, posx, posy, dst_arr, ldst, sol);
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
		while(dst_arr[0].dst < ldst)
		{
			ldst = dst_arr[0].dst;
			x = dst_arr[0].i;
			y = dst_arr[0].j;
			arrange_route(route, x, y, n);
			two_opt<<<blk, thrd>>>(route, n, posx, posy, dst_arr, ldst, sol);
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
		if(ldst < ld1)
		{
			ld1 = ldst;
			flag = 1;
			cnt++;
		}
	}
	dst = (float) ldst / 100;
	end = clock();
	tm = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\t%.2f\t%d\t%f\n",dst, cnt, tm);
	cudaFree(posx);
	cudaFree(posy);
	cudaFree(route);
	return 0;
}

