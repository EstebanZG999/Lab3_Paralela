// pipeline_chunks.c
// Producer–Consumer por chunks (no bloqueante) con stop por Isend.
// Compilar: mpicc pipeline_chunks.c -o pipeline_chunks -lm
// Uso: mpirun -np 2 ./pipeline_chunks <array_size> <chunk_size> <reps> [--csv] [--plot=gnuplot|python|none]
// Ejemplo:
//   mpirun -np 2 ./pipeline_chunks 1048576 4096 10 --csv --plot=gnuplot
//
// array_size  = número de elementos (double) en el arreglo fuente
// chunk_size  = elementos por chunk (double)
// reps        = cuántas veces recorremos todo el arreglo (para promediar)
// --csv       = imprime una línea CSV con total_s y BW efectivo
// --plot=...  = genera figura tiempo vs chunk_size si ejecutas en modo barrido externo

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

static void process_chunk(double* buf, int n) {
    // Trabajo simple sobre el chunk (simula CPU work del consumidor)
    // Evita que el compilador lo elimine: una pequeña operación numérica.
    for (int i = 0; i < n; i++) {
        buf[i] = sqrt(buf[i] * 1.000001 + 3.14159);
    }
}

// Helpers de graficado (opcionales)
static int run_gnuplot(const char* csv, const char* png) {
    const char* script =
        "set term pngcairo size 1200,700\n"
        "set output 'pipeline_tiempo.png'\n"
        "set title 'Producer-Consumer: tiempo total vs tamaño de chunk'\n"
        "set xlabel 'Tamaño de chunk (elementos double)'\n"
        "set ylabel 'Tiempo total (s)'\n"
        "set grid\n"
        "set datafile separator ','\n"
        "set logscale x 2\n"
        "plot 'resultados_pipeline.csv' using 1:2 with linespoints title 'Total s'\n";
    FILE* f = fopen("plot_pipeline.gnuplot", "w");
    if (!f) return -1;
    fputs(script, f);
    fclose(f);
    return system("gnuplot plot_pipeline.gnuplot");
}

static int run_python_plot(const char* csv, const char* png) {
    const char* py =
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "df=pd.read_csv('resultados_pipeline.csv')\n"
        "plt.figure()\n"
        "plt.plot(df['chunk_elems'], df['total_s'], marker='o')\n"
        "plt.xscale('log', base=2)\n"
        "plt.xlabel('Tamaño de chunk (elementos double)')\n"
        "plt.ylabel('Tiempo total (s)')\n"
        "plt.title('Producer-Consumer: tiempo total vs tamaño de chunk')\n"
        "plt.grid(True, which='both', linestyle='--', linewidth=0.5)\n"
        "plt.tight_layout(); plt.savefig('pipeline_tiempo.png', dpi=160)\n";
    FILE* f = fopen("plot_pipeline.py", "w");
    if (!f) return -1;
    fputs(py, f);
    fclose(f);
    return system("python3 plot_pipeline.py");
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
            fprintf(stderr, "Uso: %s <array_size> <chunk_size> <reps> [--csv] [--plot=gnuplot|python|none]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long array_size = atoll(argv[1]);   // elementos (double)
    int       chunk_size = atoi(argv[2]);    // elementos (double)
    int       reps       = atoi(argv[3]);
    int csv = 0;
    enum { PLOT_NONE, PLOT_GNUPLOT, PLOT_PYTHON } plot_mode = PLOT_NONE;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--csv") == 0) csv = 1;
        else if (strncmp(argv[i], "--plot=", 7) == 0) {
            const char* v = argv[i] + 7;
            if      (!strcmp(v,"gnuplot")) plot_mode = PLOT_GNUPLOT;
            else if (!strcmp(v,"python"))  plot_mode = PLOT_PYTHON;
            else                           plot_mode = PLOT_NONE;
        }
    }
    if (chunk_size <= 0 || array_size <= 0 || reps <= 0) {
        if (rank == 0) fprintf(stderr, "Parámetros inválidos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int TAG_DATA = 10;
    const int TAG_RESULT = 11;
    const int TAG_STOP = 99;

    // Buffers
    // Nota: contaremos elementos double (8 bytes c/u).
    double* A = NULL;       // arreglo fuente en rank0
    double* B = NULL;       // salida acumulada en rank0 (opcional)
    double* chunk_buf = (double*)xmalloc((size_t)chunk_size * sizeof(double)); // buffer de rank1

    if (rank == 0) {
        A = (double*)xmalloc((size_t)array_size * sizeof(double));
        B = (double*)xmalloc((size_t)array_size * sizeof(double));
        for (long long i = 0; i < array_size; i++) A[i] = (double)i; // datos dummy
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        // Producer: por cada repetición, enviar todos los chunks y recibir resultados.
        MPI_Request req_s, req_r;
        for (int r = 0; r < reps; r++) {
            for (long long off = 0; off < array_size; off += chunk_size) {
                int n = (int)MIN((long long)chunk_size, array_size - off);

                // No bloqueantes: envío y recepción del resultado
                MPI_Isend(&A[off], n, MPI_DOUBLE, 1, TAG_DATA, MPI_COMM_WORLD, &req_s);
                MPI_Irecv(&B[off], n, MPI_DOUBLE, 1, TAG_RESULT, MPI_COMM_WORLD, &req_r);

                // Espera de ambas (pipeline simple por chunk)
                MPI_Wait(&req_s, MPI_STATUS_IGNORE);
                MPI_Wait(&req_r, MPI_STATUS_IGNORE);
            }
        }

        // STOP: mensaje de control (0 elementos) con Isend
        MPI_Isend(NULL, 0, MPI_DOUBLE, 1, TAG_STOP, MPI_COMM_WORLD, &req_s);
        MPI_Wait(&req_s, MPI_STATUS_IGNORE);
    } else {
        // Consumer: esperar el siguiente mensaje (chunk o STOP), procesar y devolver
        MPI_Status st;
        for (;;) {
            // Espera bloqueante a cualquier mensaje del productor
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

            if (st.MPI_TAG == TAG_STOP) {
                // Consumir el STOP y salir
                MPI_Recv(NULL, 0, MPI_DOUBLE, 0, TAG_STOP, MPI_COMM_WORLD, &st);
                break;
            }

            // Es un chunk de datos
            int n = 0;
            MPI_Get_count(&st, MPI_DOUBLE, &n);
            if (n <= 0 || n > chunk_size) {
                // Sanidad: nunca debe pasar, pero evita bloqueos si hay inconsistencia
                n = (n <= 0) ? chunk_size : chunk_size;
            }

            // Recibir exactamente el payload detectado por Probe
            MPI_Recv(chunk_buf, n, MPI_DOUBLE, 0, TAG_DATA, MPI_COMM_WORLD, &st);

            // Procesar inmediatamente
            process_chunk(chunk_buf, n);

            // Devolver el resultado
            MPI_Send(chunk_buf, n, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double total = t1 - t0;

    if (rank == 0) {
        // Métrica: tiempo total y ancho de banda efectivo (ida+vuelta del chunk)
        // Bytes movidos por repetición = 2 * array_size * sizeof(double)
        double bytes_total = (double)reps * 2.0 * (double)array_size * sizeof(double);
        double MBps = (bytes_total / total) / (1024.0*1024.0);

        if (csv) {
            // CSV por ejecución individual (un chunk_size)
            // chunk_elems,total_s,MBps,array_elems,reps
            printf("%d,%.6f,%.3f,%lld,%d\n", chunk_size, total, MBps, array_size, reps);
        } else {
            printf("array=%lld doubles  chunk=%d  reps=%d\n", array_size, chunk_size, reps);
            printf("tiempo_total=%.6f s  BW_efectivo=%.3f MB/s\n", total, MBps);
        }

        // Si estás haciendo un barrido externo y acumulando a resultados_pipeline.csv,
        // puedes activar el plot automático:
        if (plot_mode != PLOT_NONE) {
            // Espera que exista resultados_pipeline.csv si haces barridos vía bash
            FILE* test = fopen("resultados_pipeline.csv", "r");
            if (test) {
                fclose(test);
                int rc = (plot_mode==PLOT_GNUPLOT) ? run_gnuplot("resultados_pipeline.csv","pipeline_tiempo.png")
                                                   : run_python_plot("resultados_pipeline.csv","pipeline_tiempo.png");
                if (rc==0) {
                    printf("Figura generada: pipeline_tiempo.png\n");
                } else {
                    fprintf(stderr, "No se pudo generar la figura automaticamente (plot rc=%d).\n", rc);
                }
            }
        }
    }

    free(chunk_buf);
    if (rank == 0) { free(A); free(B); }
    MPI_Finalize();
    return 0;
}
