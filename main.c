/*
Name: Ethan Ciavolella
Id: 916328107
Homework: #3
To Compile: Run "make" in this directory on a unix machine
To Run: mpiexec -n <Number of processes> ./GameOfLife <Board size> <Number of
generations>
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

// Uncomment this macro to replace the randomized board with a glider
// Use this with a small board size (10-20) to evaluate correctness
#define DEBUG

// Testing to see if forcing gcc to inline the functions significantly affected
// performance. It did not. #define MAYBE_INLINE // Uncomment the macro to
// disable inlining
#ifndef MAYBE_INLINE
#define MAYBE_INLINE __attribute__((always_inline)) inline
#endif

typedef uint8_t  cell_t;
typedef cell_t **board_t;

int board_size, num_generations, num_block_rows, num_tasks, task_id;

// Return current time in seconds
double gettime() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

// Update the border cells, which are not part of the simulation
// and are only there to make the board appear to wrap around
MAYBE_INLINE void clone_edges(board_t board) {

    for (int i = 1; i < num_block_rows - 1; i++) {
        board[i][0] = board[i][board_size];
        board[i][board_size + 1] = board[i][1];
    }

    if (num_tasks == 1) {
        for (int i = 0; i < board_size + 2; i++) {
            board[0][i] = board[board_size - 1][i];
            board[board_size + 1][i] = board[1][i];
        }
        return;
    }

    int dst = task_id - 1;
    if (dst < 0) {
        dst = num_tasks - 1;
    }

    cell_t *buf = malloc((2 + board_size) * sizeof(cell_t));

    MPI_Status result;
    MPI_Sendrecv(board[1], 2 + board_size, MPI_BYTE, dst, task_id, buf,
                 2 + board_size, MPI_BYTE, dst, dst, MPI_COMM_WORLD, &result);

    memcpy(board[0], buf, (2 + board_size) * sizeof(cell_t));

    dst = (task_id + 1) % num_tasks;
    MPI_Sendrecv(board[num_block_rows - 2], 2 + board_size, MPI_BYTE, dst,
                 task_id, buf, 2 + board_size, MPI_BYTE, dst, dst,
                 MPI_COMM_WORLD, &result);

    memcpy(board[num_block_rows - 1], buf, (2 + board_size) * sizeof(cell_t));
    free(buf);
}

MAYBE_INLINE uint8_t update_cell(board_t current, board_t next, int x, int y) {
    uint8_t n_neighbors = current[y - 1][x - 1];
    uint8_t has_updated = 0;

    n_neighbors += current[y - 1][x];
    n_neighbors += current[y - 1][x + 1];

    n_neighbors += current[y][x - 1];
    n_neighbors += current[y][x + 1];

    n_neighbors += current[y + 1][x - 1];
    n_neighbors += current[y + 1][x];
    n_neighbors += current[y + 1][x + 1];

    if (current[y][x]) {
        if (n_neighbors < 2) { // Living cell killed by underpopulation

            next[y][x] = 0;
            has_updated = 1;
        } else if (n_neighbors > 3) { // Living cell killed by overpopulation
            next[y][x] = 0;
            has_updated = 1;
        } else {
            next[y][x] = 1;
        }
    } else {
        if (n_neighbors == 3) { // Dead cell is resurrected
            next[y][x] = 1;
            has_updated = 1;
        } else {
            next[y][x] = 0;
        }
    }
    return has_updated;
}

// Update next's state by applying Game of Life rules to current
// Returns 1 if the board was updated, and 0 if not. This is used for early
// exit if the board state becomes finalized
MAYBE_INLINE uint8_t update(board_t current, board_t next) {
    int has_updated = 0;

    for (int i = 1; i < num_block_rows - 1; i++) {
        for (int j = 1; j < board_size + 1; j++) {
            if (update_cell(current, next, j, i)) {
                has_updated = 1;
            }
        }
    }
    clone_edges(next);

    return has_updated;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    if (argc != 3) {
        if (!task_id) {
            printf("Usage: GameOfLife <Size> <Generations>\n");
        }
        return 1;
    }

    board_size = atoi(argv[1]);
    if (board_size < 2) {
        if (!task_id) {
            printf("Board size must be greater than 1\n");
        }
        return 2;
    }

    num_generations = atoi(argv[2]);
    if (num_generations < 0) {
        if (!task_id) {
            printf("Number of generations must be at least 0\n");
        }
        return 3;
    }

    num_block_rows = 2 + board_size / num_tasks;
    if (!task_id) {
        // If the work can't be divided evenly, task 0 gets the rest of the work
        num_block_rows += board_size % num_tasks;
    }

    board_t block1 =
        malloc(num_block_rows * (2 + board_size) * sizeof(cell_t *));
    board_t block2 =
        malloc(num_block_rows * (2 + board_size) * sizeof(cell_t *));

    for (int i = 0; i < num_block_rows; i++) {
        block1[i] = malloc((board_size + 2) * sizeof(cell_t));
        block2[i] = malloc((board_size + 2) * sizeof(cell_t));
    }

    // Set up board
    srand(1337);

    // Need to call rng a bunch of times to make sure we get the same board as
    // the sequential version
    if (task_id) {
        for (int i = 0; i < board_size % num_tasks; i++) {
            for (int j = 0; j < board_size; j++) {
                rand();
            }
        }
    }
    for (int i = task_id; i > 0; i--) {
        for (int j = 0; j < board_size; j++) {
            rand();
        }
    }

    for (int i = 1; i < num_block_rows - 1; i++) {
        for (int j = 1; j < board_size + 1; j++) {
#ifdef DEBUG
            block1[i][j] = 0;
#else
            block1[i][j] = rand() % 2;
#endif
        }
    }
#ifdef DEBUG
    if (!task_id) {
        if ((board_size > 3) && (num_block_rows > 4)) {
            block1[1][2] = 1;
            block1[2][3] = 1;
            block1[3][1] = 1;
            block1[3][2] = 1;
            block1[3][3] = 1;
        } else {
            printf("Error! Could not initialize debug board\n");
        }
    }
    /*
    if (task_id == 1) {
        for (int i = 0; i < num_block_rows; i++) {
            for (int j = 0; j < board_size + 2; j++) {
                printf("%d ", (unsigned int)block1[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("Cloning edges\n");
    }
    */
#endif

    clone_edges(block1);

    /*
    if (task_id == 1) {
        printf("Done cloning edges\n");
        for (int i = 0; i < num_block_rows; i++) {
            for (int j = 0; j < board_size + 2; j++) {
                printf("%d ", (unsigned int)block1[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */

    double  before = gettime();
    int     generations_completed = 0;
    uint8_t updates = 0;

    while (generations_completed < num_generations) {
        updates |= update(block1, block2);

        if (!updates) {
            generations_completed = num_generations;
        }

        board_t temp = block1;
        block1 = block2;
        block2 = temp;

        generations_completed++;
        updates = 0;
    }

    double after = gettime();

    cell_t *buf =
        (cell_t *)malloc(num_block_rows * (board_size + 2) * sizeof(cell_t));
    if (task_id) {
        for (int i = 0; i < num_block_rows; i++) {
            memcpy(buf + i * (board_size + 2), block1[i], board_size + 2);
        }
        MPI_Gather(buf, num_block_rows * (board_size + 2), MPI_BYTE, NULL, 0,
                   MPI_BYTE, 0, MPI_COMM_WORLD);

    } else {
        printf("Completed %d generations. Runtime in seconds: %lf\n\n",
               generations_completed, after - before);

        cell_t *output_buffer = (cell_t *)malloc(
            (board_size + 2) * (board_size + 2) * sizeof(cell_t));

        for (int i = 0; i < num_block_rows; i++) {
            memcpy(buf + i * (board_size + 2), block1[i], board_size + 2);
        }
        MPI_Gather(buf, (board_size + 2) * (2 + board_size / num_tasks),
                   MPI_BYTE, output_buffer,
                   (board_size + 2) * (2 + board_size / num_tasks), MPI_BYTE, 0,
                   MPI_COMM_WORLD);

        for (int i = 2 + board_size / num_tasks; i < num_block_rows; i++) {
            for (int j = 0; j < 2 + board_size; j++) {
                output_buffer[i * (2 + board_size) + j] = block1[i][j];
            }
        }

        FILE *outfile = fopen("output.txt", "w");
        if (!outfile) {
            printf("Failed to open output file for writing\n");
        } else {
            fprintf(outfile,
                    "Completed %d generations. Runtime in seconds: %lf\n\n",
                    generations_completed, after - before);

            for (int i = 1; i < board_size + 1; i++) {
                for (int j = 1; j < board_size + 1; j++) {
                    fprintf(outfile, "%c ",
                            '0' + output_buffer[i * (board_size + 2) + j]);
                }
                fputc('\n', outfile);
            }
            fclose(outfile);
        }

        free(output_buffer);
    }

    for (int i = 0; i < num_block_rows; i++) {
        free(block1[i]);
        free(block2[i]);
    }
    free(block1);
    free(block2);
    free(buf);

    MPI_Finalize();
    return 0;
}
