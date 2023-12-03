#include <eigen3/Eigen/Dense>
#include <iostream>
#include <omp.h>

int main() {
	#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", thread_id, num_threads);
    }
}
