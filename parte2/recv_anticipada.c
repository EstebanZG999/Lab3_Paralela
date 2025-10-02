// recv_anticipada.c
// Comunicación no bloqueante: rank 1 recibe mientras calcula (MPI_Irecv + MPI_Test)
// Compilar: mpicc recv_anticipada.c -o recv_anticipada -lm
// Uso: mpirun -np 2 ./recv_anticipada <reps> <bytes> <compute_iters> [--csv] [--warmup=100]
//  Ejemplo: mpirun -np 2 ./recv_anticipada 10000 4096 200 --csv --warmup=500
//
// <reps>            = repeticiones de experimento
// <bytes>           = tamaño del mensaje
// <compute_iters>   = trabajo aritmético por ciclo de sondeo
// --csv             = imprime una línea CSV con resultados agregados
// --warmup=N        = iteraciones de calentamiento

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

// Trabajo aritmético
static inline double do_compute_step(int iters) {
    double acc = 0.0, x = 1.000001;
    for (int i = 0; i < iters; i++) {
        x = sin(x) + 1.000001;
        acc += x;
    }
    return acc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=-1, size=-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Este programa requiere -np 2\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Uso: %s <reps> <bytes>=>=8 <compute_iters> [--csv] [--warmup=100]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long reps   = atoll(argv[1]);
    long long bytes  = atoll(argv[2]);
    int compute_iters= atoi(argv[3]);
    int csv = 0;
    long long warmup = 100;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) csv = 1;
        else if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoll(argv[i]+9);
    }
    if (bytes < 8) {
        if (rank == 0) fprintf(stderr, "bytes debe ser >= 8 (8B para el entero en el header)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int tag = 123;

    char* sendbuf = (char*)xmalloc((size_t)bytes);
    char* recvbuf = (char*)xmalloc((size_t)bytes);

    // Warmup
    for (long long w = 0; w < warmup; w++) {
        if (rank == 0) {
            *(long long*)sendbuf = (long long)w; // valor cualquiera
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
        } else {
            MPI_Status st;
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &st);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Métricas agregadas
    double total_wait_s = 0.0;   // tiempo desde post Irecv hasta completarse
    long long total_compute_loops = 0; // cuántos ciclos de compute + test corrieron
    double checksum = 0.0;       // evitar que el compilador elimine el cómputo

    for (long long r = 0; r < reps; r++) {
        if (rank == 0) {
            // Preparar payload: primer 8B = entero enviado; resto, datos dummy
            *(long long*)sendbuf = (long long)(r + 1); // valor "entero" que viaja
            for (int i = 8; i < bytes; i++) sendbuf[i] = (char)((r + i) & 0xFF);

            // Para simular condiciones realistas podríamos introducir jitter aquí, pero no es necesario
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
        } else {
            // Rank 1: post no bloqueante y calcular mientras se sondea
            MPI_Request req; MPI_Status st;
            int flag = 0;

            MPI_Irecv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &req);
            double t_start = MPI_Wtime();
            long long loops = 0;

            // Bucle de solapamiento: calcular y testear
            while (!flag) {
                checksum += do_compute_step(compute_iters);
                loops++;
                MPI_Test(&req, &flag, &st);
            }
            double t_end = MPI_Wtime();

            total_wait_s += (t_end - t_start);
            total_compute_loops += loops;

            if ( (r == 0) || (r == reps/4) || (r == reps/2) || (r == (3*reps)/4) || (r == reps-1) ) {
                long long valor = *(long long*)recvbuf;
                printf("[rank1] iter=%lld valor=%lld loops_compute=%lld\n", r+1, valor, loops);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    if (rank == 0) {
        if (csv) {
            printf("rank0,%lld,%lld,%d,%lld,%.6f\n", reps, bytes, compute_iters, warmup, elapsed);
        } else {
            printf("Rank0: reps=%lld bytes=%lld compute_iters=%d warmup=%lld tiempo_total=%.6f s\n",
                   reps, bytes, compute_iters, warmup, elapsed);
        }
    } else {
        double avg_wait = total_wait_s / (double)reps;
        double avg_loops= (double)total_compute_loops / (double)reps;
        if (csv) {
            printf("rank1,%lld,%lld,%d,%lld,%.6f,%.9f,%.2f,%.3f\n",
                   reps, bytes, compute_iters, warmup, elapsed, avg_wait, avg_loops, checksum);
        } else {
            printf("Rank1: reps=%lld bytes=%lld compute_iters=%d warmup=%lld\n", reps, bytes, compute_iters, warmup);
            printf("       tiempo_total=%.6f s  wait_promedio=%.9f s  loops_promedio=%.2f  checksum=%.3f\n",
                   elapsed, avg_wait, avg_loops, checksum);
        }
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
