#include <stdio.h>
#include <time.h>
#include "mytimer.h"

static struct timespec time_start, time_end;
void timer_start(){
    clock_gettime(CLOCK_REALTIME, &time_start);
}

double timer_end(){
    clock_gettime(CLOCK_REALTIME, &time_end);
    return (time_end.tv_sec - time_start.tv_sec) + (time_end.tv_nsec - time_start.tv_nsec)/1e+9;
}