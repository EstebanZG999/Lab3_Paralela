// pingpong_sweep.c
// Barre tamaños, mide RTT (ida+vuelta), escribe CSV y genera gráfica.
// Compilar:  mpicc pingpong_sweep.c -o pingpong_sweep
// Usar:      mpirun -np 2 ./pingpong_sweep [--warmup=200] [--reps=10000] [--plot=gnuplot|python|none]
// Salidas:   resultados_pingpong.csv  y  pingpong_rtt.png  (si --plot != none)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

// ----- Graficado con Gnuplot -----
static int run_gnuplot(const char* csv, const char* png) {
    const char* script =
        "set term pngcairo size 1200,700\n"
        "set output 'pingpong_rtt.png'\n"
        "set title 'Ping-Pong MPI: Tamano de mensaje vs RTT'\n"
        "set xlabel 'Tamano de mensaje (bytes)'\n"
        "set ylabel 'RTT promedio (\\265s)'\n"   // \265 = µ
        "set grid\n"
        "set datafile separator ','\n"
        "set logscale x 2\n"
        "set xrange [1:*]\n"
        "set key left top\n"
        "plot 'resultados_pingpong.csv' using 1:($4*1e6) with linespoints title 'RTT'\n";

    FILE* gp = fopen("plot_pingpong.gnuplot", "w");
    if (!gp) return -1;
    fputs(script, gp);
    fclose(gp);

    int rc = system("gnuplot plot_pingpong.gnuplot");
    return rc;
}

// ----- Graficado con Python (pandas + matplotlib) -----
static int run_python_plot(const char* csv, const char* png) {
    const char* py =
        "import sys, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "csv='resultados_pingpong.csv'\n"
        "out='pingpong_rtt.png'\n"
        "df=pd.read_csv(csv)\n"
        "df['RTT_us']=df['RTT_s']*1e6\n"
        "plt.figure()\n"
        "plt.plot(df['bytes'], df['RTT_us'], marker='o')\n"
        "plt.xscale('log', base=2)\n"
        "plt.xlabel('Tamaño de mensaje (bytes)')\n"
        "plt.ylabel('RTT promedio (µs)')\n"
        "plt.title('Ping-Pong MPI: Tamaño de mensaje vs RTT')\n"
        "plt.grid(True, which='both', linestyle='--', linewidth=0.5)\n"
        "plt.tight_layout()\n"
        "plt.savefig(out, dpi=160)\n";

    FILE* f = fopen("plot_pingpong.py", "w");
    if (!f) return -1;
    fputs(py, f);
    fclose(f);

    int rc = system("python3 plot_pingpong.py");
    return rc;
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

    long long warmup = 200;
    long long reps_base = 10000;
    enum { PLOT_GNUPLOT, PLOT_PYTHON, PLOT_NONE } plot_mode = PLOT_GNUPLOT;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoll(argv[i]+9);
        else if (strncmp(argv[i], "--reps=", 7) == 0) reps_base = atoll(argv[i]+7);
        else if (strncmp(argv[i], "--plot=", 7) == 0) {
            const char* v = argv[i]+7;
            if (strcmp(v,"gnuplot")==0) plot_mode = PLOT_GNUPLOT;
            else if (strcmp(v,"python")==0) plot_mode = PLOT_PYTHON;
            else if (strcmp(v,"none")==0)   plot_mode = PLOT_NONE;
        }
    }

    const long long sizes[] = { 1, 8, 64, 512, 4096, 32768, 262144, 1048576 };
    const int NS = (int)(sizeof(sizes)/sizeof(sizes[0]));

    FILE* csv = NULL;
    if (rank == 0) {
        csv = fopen("resultados_pingpong.csv", "w");
        if (!csv) {
            fprintf(stderr, "No pude crear resultados_pingpong.csv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(csv, "bytes,reps,warmup,RTT_s,lat_s,MBps,Mbps\n");
    }

    const int tag = 42;

    for (int k = 0; k < NS; k++) {
        long long bytes = sizes[k];

        long long reps = reps_base;
        if (bytes >= 32768)  reps = reps_base/1;    // 10k por defecto
        if (bytes >= 262144) reps = reps_base/2;    // 5k
        if (bytes >= 1048576)reps = reps_base/5;    // 2k
        if (reps < 2000)     reps = 2000;

        char* sendbuf = (char*)xmalloc((size_t)bytes);
        char* recvbuf = (char*)xmalloc((size_t)bytes);
        for (long long i = 0; i < bytes; i++) sendbuf[i] = (char)(i % 251);

        MPI_Barrier(MPI_COMM_WORLD);
        for (long long r = 0; r < warmup; r++) {
            if (rank == 0) {
                MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
                MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (long long r = 0; r < reps; r++) {
            if (rank == 0) {
                MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
                MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(recvbuf, (int)bytes, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
            }
        }
        double t1 = MPI_Wtime();
        double elapsed = t1 - t0;
        double rtt_avg  = elapsed / (double)reps;
        double lat_one  = rtt_avg / 2.0;
        double bw_Bps   = (2.0 * (double)bytes) / rtt_avg;
        double bw_MBps  = bw_Bps / (1024.0*1024.0);
        double bw_Mbps  = (bw_Bps * 8.0) / 1.0e6;

        if (rank == 0) {
            printf("[PP] bytes=%lld reps=%lld RTT=%.9f s lat=%.9f s BW=%.3f MB/s\n",
                   bytes, reps, rtt_avg, lat_one, bw_MBps);
            fprintf(csv, "%lld,%lld,%lld,%.9f,%.9f,%.3f,%.3f\n",
                    bytes, reps, warmup, rtt_avg, lat_one, bw_MBps, bw_Mbps);
            fflush(csv);
        }

        free(sendbuf);
        free(recvbuf);
    }

    if (rank == 0) {
        fclose(csv);
        int rc = 0;
        if (plot_mode == PLOT_GNUPLOT) {
            rc = run_gnuplot("resultados_pingpong.csv", "pingpong_rtt.png");
            if (rc != 0) {
                fprintf(stderr, "Gnuplot fallo (rc=%d). Prueba instalar gnuplot-nox o usa --plot=python.\n", rc);
            } else {
                printf("Grafica generada: pingpong_rtt.png (gnuplot)\n");
            }
        } else if (plot_mode == PLOT_PYTHON) {
            rc = run_python_plot("resultados_pingpong.csv", "pingpong_rtt.png");
            if (rc != 0) {
                fprintf(stderr, "Python plot fallo (rc=%d). Revisa tener python3 + pandas + matplotlib, o usa --plot=gnuplot.\n", rc);
            } else {
                printf("Grafica generada: pingpong_rtt.png (python)\n");
            }
        } else {
            printf("CSV generado: resultados_pingpong.csv (sin grafica por --plot=none)\n");
        }
    }

    MPI_Finalize();
    return 0;
}
