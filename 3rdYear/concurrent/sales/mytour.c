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
#include <math.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include "mytour.h"

///////////////////////////////////////////////////////
/*
Things that i've tried:
Trying to reduce the number of loops that the innner for goes through
by having each loop deal with two at a time, this turned out to just increase
the time required as the increased number of comparisons required for this 
actually increased the number of instructions.

Turned to simply reducing the number of distance calls by creating a variable
that holds the value returned by distance, this can both dramatically increase or decrease
the speed, usually it decreases. This is because 
*/
///////////////////////////////////////////////////////

float square(float x) {
  return x*x;
}

float distance(const point cities[], int i, int j) {
  return sqrt(square(cities[i].x-cities[j].x)+
        square(cities[i].y-cities[j].y));
}

void my_tour(const point cities[], int tour[], int ncities) {
  // its the same as simple_find_tour until the for loops
  int i, j;
  char *visited = alloca(ncities);
  int ThisPt, ClosePt = 0;
  float CloseDist, dist;
  int endtour = 0;

  for (i = 0; i < ncities; i++) 
    visited[i] = 0;

  ThisPt = ncities - 1;
  visited[ncities - 1] = 1;
  tour[endtour++] = ncities - 1;

  // only starting using the multi threaded code past 5000 cities, this is because the overhead required to use threads will make code before this often be slower than
  // none threaded implementations
  if (ncities < 5000) {
    // This is the non-threaded implementation: its the same as the simple_find_tour except for our adding the dist variable, this variable stores our call to
    // distance, this reduces our calls to distance and speeds up the program, this will be used in the multi threaded implementation too, the improvement is
    // very small and will sometimes still have our code be slightly slower
    for (i=1; i<ncities; i++) {
      CloseDist = DBL_MAX;
      for (j=0; j<ncities-1; j++) {
        if (!visited[j]) {
          if ((dist = distance(cities, ThisPt, j)) < CloseDist) {
            CloseDist = dist;
            ClosePt = j;
          }
        }
      }
      tour[endtour++] = ClosePt;
      visited[ClosePt] = 1;
      ThisPt = ClosePt;
    }
  } else {
    // This is the multi-threaded implementation, here, we create our threads before entering the for loop and make the dist variable private for all of them.
    for ( i = 1; i < ncities; i++) {
      CloseDist = DBL_MAX;
      #pragma omp parallel private(dist)
      {
        int localClosePt = 0;
        float localCloseDist = CloseDist;
        // Find the local closest point of all the iterations that a thread has, we don't need to wait for all the threads to finish as we aren't doing anything
        // in the rest of the parallel code block that requires the threads be synchronized
        #pragma omp for nowait
        for (j = 0; j < ncities - 1; j++){
          if (!visited[j]) {
            dist = distance(cities, ThisPt, j);
            if (dist < localCloseDist) {
              localCloseDist = dist;
              localClosePt = j;
            }
          }
        }
        // once a thread has finished finding it's local closest point, see if that point is closer than the current global closest point, if it is
        // set CloseDist and ClosePt to now be the threads local point and distance
        // this must be critical as the code within the if statement affects access to the if statement.
        #pragma omp critical
        {
          if (localCloseDist < CloseDist) {
            CloseDist = localCloseDist;
            ClosePt = localClosePt;
          }
        }
      }
      tour[endtour++] = ClosePt;
      visited[ClosePt] = 1;
      ThisPt = ClosePt;
    }
  } 
}