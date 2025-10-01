// token_ring_simple.c
// Implementa un Token Ring básico con MPI_Send y MPI_Recv.
// Uso: mpirun -np 6 ./token_ring_simple <vueltas> <bytes>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 4) {
        if (rank == 0) fprintf(stderr, "Este ejercicio pide -np > 4\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Uso: %s <vueltas> <bytes>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long vueltas = atoll(argv[1]);
    long long nbytes  = atoll(argv[2]);

    const int tag = 77;
    int src = (rank - 1 + size) % size;  // anterior en el anillo
    int dst = (rank + 1) % size;         // siguiente en el anillo

    char* sendbuf = (char*)xmalloc((size_t)nbytes);
    char* recvbuf = (char*)xmalloc((size_t)nbytes);
    memset(sendbuf, (unsigned char)(rank & 0xFF), (size_t)nbytes);

    long long vueltas_realizadas = 0;

    // Solo rank 0 arranca el token
    if (rank == 0) {
        *(long long*)sendbuf = 0; // contador de vueltas en el token
        MPI_Send(sendbuf, (int)nbytes, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (;;) {
        MPI_Status st;
        // Cada rank recibe del anterior
        MPI_Recv(recvbuf, (int)nbytes, MPI_BYTE, src, tag, MPI_COMM_WORLD, &st);

        // Rank 0 incrementa el contador cuando el token completa una vuelta
        if (rank == 0) {
            long long* p = (long long*)recvbuf;
            (*p)++;
            vueltas_realizadas = *p;
        }

        // Copiar el token y enviarlo al siguiente
        memcpy(sendbuf, recvbuf, (size_t)nbytes);
        MPI_Send(sendbuf, (int)nbytes, MPI_BYTE, dst, tag, MPI_COMM_WORLD);

        // Condición de salida
        long long fin = 0;
        if (rank == 0 && vueltas_realizadas >= vueltas) fin = 1;
        MPI_Bcast(&fin, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        if (fin) break;
    }

    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("[TokenRing-simple] np=%d, vueltas=%lld, bytes=%lld, tiempo=%.6f s\n",
               size, vueltas, nbytes, t1 - t0);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
