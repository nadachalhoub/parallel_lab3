#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function prototypes
void merge_sort(int *array, int left, int right);
void merge(int *array, int left, int mid, int right);

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 1000000; // Size of the array
    int *array = NULL, *sub_array = NULL;
    int sub_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Master process: Initialize array
        array = (int *)malloc(n * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100000; // Random integers
        }
    }

    // Determine the size of each sub-array
    sub_size = n / size;
    sub_array = (int *)malloc(sub_size * sizeof(int));

    // Scatter the array to all processes
    MPI_Scatter(array, sub_size, MPI_INT, sub_array, sub_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its sub-array using OpenMP
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(sub_array, 0, sub_size - 1);
    }

    // Gather sorted sub-arrays back to the master process
    MPI_Gather(sub_array, sub_size, MPI_INT, array, sub_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Master process performs final merge
        int *sorted_array = (int *)malloc(n * sizeof(int));
        int step = sub_size;

        // Parallel merge with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < size - 1; i++) {
            merge(array, 0, (i + 1) * step - 1, (i + 2) * step - 1);
        }

        free(sorted_array);
    }

    free(sub_array);
    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();
    return 0;
}

// Recursive merge sort
void merge_sort(int *array, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp task
        merge_sort(array, left, mid);

        #pragma omp task
        merge_sort(array, mid + 1, right);

        #pragma omp taskwait
        merge(array, left, mid, right);
    }
}

// Merge function
void merge(int *array, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *left_array = (int *)malloc(n1 * sizeof(int));
    int *right_array = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) {
        left_array[i] = array[left + i];
    }
    for (int i = 0; i < n2; i++) {
        right_array[i] = array[mid + 1 + i];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (left_array[i] <= right_array[j]) {
            array[k++] = left_array[i++];
        } else {
            array[k++] = right_array[j++];
        }
    }

    while (i < n1) {
        array[k++] = left_array[i++];
    }
    while (j < n2) {
        array[k++] = right_array[j++];
    }

    free(left_array);
    free(right_array);
}
