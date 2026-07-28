#include"stdio.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include"math.h"
#include <ctype.h>
#include <assert.h>

/* Euclidean distance calculation */
long distD(int i, int j, float *x, float*y)
{
	float dx = x[i] - x[j];
	float dy = y[i] - y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}


/* Initial solution construction using NN */
long nn_route(int *route, long n, float *posx, float*posy)
{
	route[0] = 0;
	int k = 1, i = 0, j;
	float min;
	int minj, mini, count = 1, flag = 0;
	long ltcy = 0;
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

/* swap move evaluation function */
long swap(int *rt, long n, float *posx, float*posy, long cost)
{
	int i, j, min_i, min_j, tmp, flag = 1, id, cnt = 0;
	long change, minchange = 0;
	while(flag)
	{
		flag = 0;
		minchange = 0;
		for(i = 1; i < n-1; i++)
		{
			for(j = i+1; j < n; j++)
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
	 			else if(j < n-1)
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
				if(change <minchange)
				{
					minchange = change;
					min_i = i;
					min_j = j;
					id = i * (n - 1) + (j - 1) - i * (i + 1) / 2;	
				}
			}
		}
		if(minchange < 0)
		{
			cost += minchange;
			tmp = rt[min_i];
			rt[min_i] = rt[min_j];
			rt[min_j] = tmp;
			flag = 1;
			cnt++;
		}
	
	}
	return cost;
}
/* Sets a new solution using i, j pair */
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
/* Calculate latency of a solution for applying two-opt move on i, j pair*/
long get_route_dst(int*route, float *posx, float *posy, int i, int j, int n)
{
	long ltcy = 0;
	long d1 = 0, d2 = 0, d3 = 0;
	int x, y, z;
	for(x = 0, y =1; y <=i; x++, y++)
		d1 += (n - y) * distD(route[x], route[y], posx, posy);
	for(y = j + 1, x = i + 1; y < n; x = y, y++)
		d2 += (n - y) * distD(route[x], route[y], posx, posy);
	for( x = i, y = j, z = i; y > i; x = y, y--, z++)
		d3 += (n - z -1) * distD(route[x], route[y], posx, posy);
	ltcy = d1 + d2 + d3;
	return ltcy;
}
/* Prints current solution */
void print_route(int *rt, int n)
{
	int i;
	printf("\nnew route\n");
	for(i = 0; i < n; i++)
		printf("%d, ", rt[i]);
	printf("\n");
}
/* two-opt move evaluation function */
long two_opt(int *route, long n, float *posx, float*posy, long cost)
{
	int i, j, min_i, min_j, tmp, flag = 1, cnt = 0;
	long change, minchange = 0, new_cost, tmp_cost = cost;
	int *new_route, *tmp_route;
	new_route = (int*)malloc(sizeof(int)*n);
	while(flag)
	{
		flag = 0;
		tmp_cost = cost;
		for(i = 1; i < n-1; i++)
		{
			for(j = i+2; j < n; j++)
			{
				new_cost = get_route_dst(route, posx, posy, i, j, n);
				if(new_cost < tmp_cost)
				{
					tmp_cost = new_cost;
					min_i = i;
					min_j = j;
				}
			}
		}
		if(tmp_cost < cost)
		{
			cost = tmp_cost;
			arrange_route(route, min_i, min_j, n);
			flag = 1;
			cnt++;
		}
	}
	return cost;
}
/* verify the contructed solution is valid or not */
void route_checker(int *route, int n)
{
	int i, j, *v, flag = 0;
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
	float *px, *py,tm;
	char str[256];  
	int *r;
	long dst, sol, d1, tid=0, ltcy;
	int i, j, k, intl, count, *route, *v, flag;
	
	clock_t start,end,start1,end1;

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
		n = i; n++;
		route = (int *)malloc(sizeof(int) * n);
		fscanf(f, "%s\n", str);
		while (strcmp(str, "y-Coor") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
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
		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
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
	route = (int *)malloc(sizeof(int) * n);
	dst = nn_route(route, n, posx, posy);
	printf("%ld\t",dst);
	route_checker(route, n);
	cnt = 1;
	flag = 1;
	while(flag)
	{
		flag = 0;
		d1 = swap(route, n, posx, posy, dst); 
		d1 = two_opt(route, n, posx, posy, d1); 
		if(dst > d1)
		{
			flag = 1;
			dst = d1;
			cnt++;
		}
	}
	end = clock();
	tm = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%ld\t%d\t%f\n", dst, cnt, tm);
	free(route);
	free(posx);
	free(posy);
	return 0;
}

