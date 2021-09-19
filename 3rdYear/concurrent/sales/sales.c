/*
To run: gcc -fopenmp sales.c mytour.c -lm

Firstly I looked at improving the algorithm, the first way I looked to do this was to reduce the number of loops the second for loop went through.
I went about this by trying to run multiple iterations in a single iteration, inspired somewhat by vector programming, while it did not occur to me
at the time, this is illogical, as you are still running the same amount of instructions, just now, you are doing half the number of loops, but twice the
amount of instructions per loop. However, it was even worse than this, as the logic required to run multiple at the same time ended up making this
approach even slower than the base code. My second attempt at improving the algorithm is much more basic and quite obvious, in the provided code, 
the dist function is called twice, the obvious improvement is to assign the value of dist to a variable instead of calling it twice.

After implementing this, I went to check my progress by adding a timer to the check_tour function, at the simple_find_tour call, this is where I met something
interesting, that is possibly just an anomaly of my machine, this code logically should be faster, even marginally, however, it is in reality, only often faster.
On some iterations, it will be slower, but on most faster, I though this was fine, there must be some logic to why this is occuring, however, not 30 minutes later,
a new problem arose, my code was all of a sudden significantly slower, by an order of magnitude, I went about attempting to figure out what was wrong with it
and found that, if I replace the simple_find_tour in the check_tour function with my_tour or an equivalent, then all of a sudden my own code is an order of magnitude
faster than my own code, only called in a different place. While this is very interesting, I did not have the time to figure out why this was the case, and with
this somehow fixing itself again, not an hour later, I decided to leave it be and simply report it here. I can report that my code, with only the algorithm
improvement supplied above, is usually 10000 microseconds faster that the base code although is somethimes is a few hundred slower, when run with 10000 cities.

Next I looked to implement multithreaded capabilies to my code, to do this, I looked to turn the second for loop into a parallel for loop, however, I found that I
couldn't get the right answer with this, I then realised that I had forgotten to make sure that only one thread was in the if statement that changes CloseDist at
a time, likely meaning the problem was simply to do with multiple threads changing CloseDist at once, and so CloseDist becomes non-deterministic. To combat this
my approach was to create all of the threads before entering the for loop, then have each thread figure out of its iterations, what is its closest point, and
then after the loop, there will be a critical section that will figure out if its local closest point is smaller than the global closest point. Not only does this
solve our issue of multiple threads entering the if statement at once meaning the last thread that to finish executing code will set the closest point, irrespective
of what the actual closest point is, it also avoids the much less efficient improvement of having the critical after running dist(), which means that each thread will
stall within the for loop, and substanially limiting the possible gains that we could achieve. On my machine, we start to see significant improvements at 10000 cities,
at which point the new code is always faster, with the old code being almost 6 times slower.

I looked at implementing vector programming, but couldn't find a way in which it would realistically fit into the program without its addition being so convoluted that
it would simply bring about incredibly difficult to fix bugs.
*/

#include <alloca.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include "sales.h"
#include "mytour.h"

const int DEBUG = 0;

float sqr(float x)
{
  return x*x;
}

float dist(const point cities[], int i, int j) {
  return sqrt(sqr(cities[i].x-cities[j].x)+
	      sqr(cities[i].y-cities[j].y));
}

void simple_find_tour(const point cities[], int tour[], int ncities)
{
  int i,j;
  char *visited = alloca(ncities);
  int ThisPt, ClosePt=0;
  float CloseDist;
  int endtour=0;
  
  for (i=0; i<ncities; i++)
    visited[i]=0;
  ThisPt = ncities-1;
  visited[ncities-1] = 1;
  tour[endtour++] = ncities-1;
  
  for (i=1; i<ncities; i++) {
    CloseDist = DBL_MAX;
    for (j=0; j<ncities-1; j++) {
      if (!visited[j]) {
	if (dist(cities, ThisPt, j) < CloseDist) {
	  CloseDist = dist(cities, ThisPt, j);
	  ClosePt = j;
	}
      }
    }
    tour[endtour++] = ClosePt;
    visited[ClosePt] = 1;
    ThisPt = ClosePt;
  }
}

/* write the tour out to console */
void write_tour(int ncities, point * cities, int * tour)
{
  int i;
  float sumdist = 0.0;

  /* write out the tour to the screen */
  printf("%d\n", tour[0]);
  for ( i = 1; i < ncities; i++ ) {
    printf("%d\n", tour[i]);
    sumdist += dist(cities, tour[i-1], tour[i]);
  }
  printf("sumdist = %f\n", sumdist);
}

/* write out an encapsulated postscript file of the tour */
void write_eps_file(int ncities, point *cities, int *tour)
{
  FILE *psfile;
  int i;

  psfile = fopen("sales.eps","w");
  fprintf(psfile, "%%!PS-Adobe-2.0 EPSF-1.2\n%%%%BoundingBox: 0 0 300 300\n");
  fprintf(psfile, "1 setlinejoin\n0 setlinewidth\n");
  fprintf(psfile, "%f %f moveto\n",
	  300.0*cities[tour[0]].x, 300.0*cities[tour[0]].y);
  for (i=1; i<ncities; i++) {
    fprintf(psfile, "%f %f lineto\n",
	    300.0*cities[tour[i]].x, 300.0*cities[tour[i]].y);
  }
  fprintf(psfile,"stroke\n");
}

/* create a random set of cities */
void initialize_cities(point * cities, int ncities, unsigned seed)
{
  int i;

  srandom(seed);
  for (i=0; i<ncities; i++) {
    cities[i].x = ((float)(random()))/(float)(1U<<31);
    cities[i].y = ((float)(random()))/(float)(1U<<31);
  }
}

int check_tour(const point *cities, int * tour, int ncities)
{
  int * tour2 = malloc(ncities*sizeof(int));
  int i;
  int result = 1;

  simple_find_tour(cities,tour2,ncities);

  for ( i = 0; i < ncities; i++ ) {
    if ( tour[i] != tour2[i] ) {
      result = 0;
    }
  }
  free(tour2);
  return result;
}

void call_student_tour(const point *cities, int * tour, int ncities)
{
  my_tour(cities, tour, ncities);
}

int main(int argc, char *argv[])
{
  int i, ncities;
  point *cities;
  int *tour;
  int seed;
  int tour_okay;
  struct timeval start_time, stop_time;
  long long compute_time;
  

  if (argc!=2) {
    fprintf(stderr, "usage: %s <ncities>\n", argv[0]);
    exit(1);
  }

  /* initialize random set of cities */
  ncities = atoi(argv[1]);
  cities = malloc(ncities*sizeof(point));
  tour = malloc(ncities*sizeof(int));
  seed = 3656384L % ncities;
  initialize_cities(cities, ncities, seed);

  /* find tour through the cities */
  gettimeofday(&start_time, NULL);
  call_student_tour(cities,tour,ncities);
  gettimeofday(&stop_time, NULL);
  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to find tour: %lld microseconds\n", compute_time);

  /* check that the tour we found is correct */
  tour_okay = check_tour(cities,tour,ncities);
  if ( !tour_okay ) {
    fprintf(stderr, "FATAL: incorrect tour\n");
  }
  
  /* write out results */
  if ( DEBUG ) {
    write_eps_file(ncities, cities, tour);
    write_tour(ncities, cities, tour);
  }

  free(cities);
  free(tour);
  return 0;
}
