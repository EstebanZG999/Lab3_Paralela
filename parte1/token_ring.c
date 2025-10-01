
// token_ring.c
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
            fprintf(stderr, "Uso: %s <vueltas> <bytes> [--mode=sendrecv|--mode=pair]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long vueltas = atoll(argv[1]);
    long long nbytes  = atoll(argv[2]);
    int use_sendrecv = 1; // por defecto

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mode=pair") == 0)         use_sendrecv = 0;
        else if (strcmp(argv[i], "--mode=sendrecv") == 0) use_sendrecv = 1;
    }

    if (nbytes < 8) {
        if (rank == 0) fprintf(stderr, "Para contar VUELTAS reales se requieren al menos 8 bytes (header de 8B).\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int tag = 77;
    int src = (rank - 1 + size) % size;      // anterior
    int dst = (rank + 1) % size;             // siguiente

    char* sendbuf = (char*)xmalloc((size_t)nbytes);
    char* recvbuf = (char*)xmalloc((size_t)nbytes);
    memset(sendbuf, 0, (size_t)nbytes);
    memset(recvbuf, 0, (size_t)nbytes);

    if (rank == 0) {
        *(long long*)sendbuf = 0;   // token válido con contador=0
    } else {
        *(long long*)sendbuf = -1;  // aún no tengo token válido
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    long long vueltas_realizadas = 0;

    for (;;) {
        MPI_Status st;

        if (use_sendrecv) {
            // Sendrecv
            MPI_Sendrecv(sendbuf, (int)nbytes, MPI_BYTE, dst, tag,
                         recvbuf, (int)nbytes, MPI_BYTE, src, tag,
                         MPI_COMM_WORLD, &st);
        } else {
            // Send + Recv pareado
            if ((rank % 2) == 0) {
                MPI_Send(sendbuf, (int)nbytes, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
                MPI_Recv(recvbuf, (int)nbytes, MPI_BYTE, src, tag, MPI_COMM_WORLD, &st);
            } else {
                MPI_Recv(recvbuf, (int)nbytes, MPI_BYTE, src, tag, MPI_COMM_WORLD, &st);
                MPI_Send(sendbuf, (int)nbytes, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
            }
        }

        // Propagar el payload completo
        memcpy(sendbuf, recvbuf, (size_t)nbytes);

        // Leer header del token que llegó
        long long in = *(long long*)recvbuf;

        if (rank == 0 && in >= 0) {
            in += 1;                       // una vuelta más
            vueltas_realizadas = in;       // actualizar progreso
        }

        // Escribir el contador actualizado en el header que saldrá
        *(long long*)sendbuf = in;

        long long fin = 0;
        if (rank == 0 && vueltas_realizadas >= vueltas) fin = 1;
        MPI_Bcast(&fin, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        if (fin) break;
    }

    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("[TokenRing-%s] np=%d, vueltas=%lld, bytes=%lld, tiempo=%.6f s\n",
               use_sendrecv ? "sendrecv" : "pair",
               size, vueltas, nbytes, t1 - t0);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
