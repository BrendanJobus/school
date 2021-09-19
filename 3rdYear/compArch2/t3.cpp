#include <iostream>
#include <conio.h>
#include <chrono>
#include <time.h>

using namespace std;
using namespace chrono;

int NUM_RUNS = 100000;
int REGISTER_WINDOWS = 6;

int calls = 0;
int depth = 0;
int maxDepth = 0;

int overflows = 0;
int underflows = 0;

int windowsInUse = 0;

void overflowCheck() {
    calls++;
    depth++;
    if(depth > maxDepth) {
        maxDepth = depth;
    }
    if(windowsInUse == REGISTER_WINDOWS) {
        overflows++;
    }
    else {
        windowsInUse++;
    }
}

void underflowCheck() {
    depth--;
    if(windowsInUse == 2) {
        underflows++;
    }
    else {
        windowsInUse--;
    }
}

int compute_pascalWithChecks(int row, int position) {
    overflowCheck();
    int ret;
    if (position == 1) {
        ret = 1;
    }
    else if (position == row) {
        ret = 1;
    }
    else {
        ret = compute_pascalWithChecks(row - 1, position) + compute_pascalWithChecks(row - 1, position - 1);
    }
    underflowCheck();
    return ret;
}

int compute_pascal(int row, int position) {
    if(position == 1) {
        return 1;
    }
    else if(position == row) {
        return 1;
    }
    else {
        return compute_pascal(row - 1, position) + compute_pascal(row - 1, position - 1);
    }
}

int main(int argc, char* argv[]) {
    compute_pascalWithChecks(30, 20);
    printf("Number of Calls: %d\nMaximum Register Windows Depth: %d\nOverflows: %d\nUnderflows: %d", calls, maxDepth, overflows, underflows);

    volatile int x = 30;
    int i = 0;
    auto start = high_resolution_clock::now();
    while(i < NUM_RUNS) {
        compute_pascal(x, 20);
        i++;
    }

    auto end = high_resolution_clock::now();

    long double dur = duration_cast<nanoseconds>(end - start).count();

    cout << "Took " << dur << " nanoseconds\n";

    double time = dur / (double)NUM_RUNS;

    cout << "Average " << time << " nanoseconds\n";

    return 0;
}