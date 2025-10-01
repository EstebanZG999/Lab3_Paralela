// pingpong.c
}// mpicc pingpong.c -o pingpong
// Uso: mpirun -np 2 ./pingpong <repeticiones> <bytes> [--csv] [--warmup=10]
// Ejemplo: mpirun -np 2 ./pingpong 10000 1024 --csv --warmup=100

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc fallo\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = -1, world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (world_rank == 0) fprintf(stderr, "Este programa requiere -np 2\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc < 3) {
        if (world_rank == 0) {
            fprintf(stderr, "Uso: %s <repeticiones> <bytes> [--csv] [--warmup=10]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long reps = atoll(argv[1]);
    long long bytes = atoll(argv[2]);
    int csv = 0;
    long long warmup = 10;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) csv = 1;
        else if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoll(argv[i] + 9);
    }

    if (reps <= 0 || bytes <= 0) {
        if (world_rank == 0) fprintf(stderr, "repeticiones y bytes deben ser > 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Buffer
    char* sendbuf = (char*)xmalloc((size_t)bytes);
    char* recvbuf = (char*)xmalloc((size_t)bytes);
    for (long long i = 0; i < bytes; i++) sendbuf[i] = (char)(i % 251);

    const int tag = 42;
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    for (long long i = 0; i < warmup; i++) {
        if (world_rank == 0) {
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Medición
    for (long long i = 0; i < reps; i++) {
        if (world_rank == 0) {
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
        }
    }

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0; // tiempo total de reps RTT

    if (world_rank == 0) {
        // Promedio por RTT
        double rtt_avg = elapsed / (double)reps;                 // segundos por ida-vuelta
        double latency_one_way = rtt_avg / 2.0;                  // latencia aprox. una vía
        // Ancho de banda efectivo: se transfieren 2*bytes por RTT (ida + vuelta)
        double bandwidth = (2.0 * (double)bytes) / rtt_avg;      // bytes/seg
        double bandwidth_MBps = bandwidth / (1024.0 * 1024.0);   // MiB/s
        double bandwidth_Mbps = (bandwidth * 8.0) / 1.0e6;       // Mbit/s

        if (csv) {
            // CSV: bytes,reps,warmup,RTT_promedio_s,latencia_s,MBps,Mbps
            printf("%lld,%lld,%lld,%.9f,%.9f,%.3f,%.3f\n",
                   bytes, reps, warmup, rtt_avg, latency_one_way, bandwidth_MBps, bandwidth_Mbps);
        } else {
            printf("Ping-Pong bloqueante\n");
            printf("bytes=%lld, reps=%lld, warmup=%lld\n", bytes, reps, warmup);
            printf("Tiempo total = %.6f s\n", elapsed);
            printf("RTT promedio = %.9f s\n", rtt_avg);
            printf("Latencia (aprox una vía) = %.9f s (%.3f us)\n", latency_one_way, latency_one_way * 1e6);
            printf("BW efectivo = %.3f MB/s (%.3f Mbit/s)\n", bandwidth_MBps, bandwidth_Mbps);
        }
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
