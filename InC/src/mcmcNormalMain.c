#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <mcmcNormal.h>

#define PROP 1
#define POST 2

int main(int argc,char *argv[])
{
  int retval;
  int iterAll, iterJ, dim;
  int world_rank, world_size;
  double starttime_setup;
  double startvalue;
  double *thetaCan, *thetaCurr, posteriorCurr, posteriorCan, qCurr, qCan;
  double acceptlevel, acceptUniform;
  double *KovMatProposal, *KovMatPosterior;
  unsigned long seed;
  gsl_rng *gslrng;

  // init MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  starttime_setup = MPI_Wtime();

  /* get inputs */
  // Constant for MCMC loop
  if (argc>1) {
    iterAll = atoi(argv[1]);
  } else {
    printf("A constant (int) for # of iterations is needed.\nEnd.\n");
    MPI_Finalize();
    return(1);
  }
  // dimension
  if (argc>2) {
    dim = atoi(argv[2]);
  } else {
    printf("No dimension 'dim' is defined. We work with dim=20.\n");
    dim = 20;
  }
  // startvalue
  if (argc>3) {
    startvalue = atoi(argv[3]);
  } else {
    printf("No startvalue 'start' is defined. We work with start=2.0.\n");
    startvalue = 2.0;
  }

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // Random number generator initialisieren
  gslrng = gsl_rng_alloc(gsl_rng_taus2);
  seed   = getSeed();            // get a seed based on current time. default seed = 0.
  gsl_rng_set(gslrng, seed);     // set a different seed for the rng.

  // Kovarianzmatrizen, damit sie nicht in jeder Iteration neu aufgebaut werden muessen
  KovMatProposal = (double *) calloc(dim, sizeof(double));
  retval = getKovMat(KovMatProposal, PROP, dim);
  KovMatPosterior = (double *) calloc(dim, sizeof(double));
  retval = getKovMat(KovMatPosterior, POST, dim);

  // Startpunkt
  thetaCan  = (double *) calloc(dim, sizeof(double));
  thetaCurr = (double *) calloc(dim, sizeof(double));
  retval = getStarted(startvalue, dim, thetaCurr, &posteriorCurr); // Startpunkt und dessen Werte setzen

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // loop
  for (int iterJ = 0; iterJ < iterAll; iterJ++) {
    // Proposal
    retval = getProposal(gslrng, dim, thetaCurr, thetaCan, &qCurr, &qCan);
    printf("candid_%3d\t",iterJ);
    for (int i = 0; i < dim; i++) { printf("%.2f\t",thetaCan[i]); }

    // Posterior
    retval = getPosterior(thetaCan, dim, &posteriorCan);
    // Akzeptanzlevel
    retval =  getAcceptancelevel(&posteriorCan, &posteriorCurr, &qCan, &qCurr, &acceptlevel);
    // AccRej
    acceptUniform = gsl_rng_uniform(gslrng);
    printf("%.3f < %.3f\t",acceptUniform, acceptlevel);
    if (acceptUniform < acceptlevel) {
      printf("YES\n");
      for (int i = 0; i < dim; i++) { thetaCurr[i] = thetaCan[i]; }
      posteriorCurr = posteriorCan;
    } else {
      printf("no\n");
    }
  }

  // free memory
  free(KovMatProposal);
  free(thetaCan);
  free(thetaCurr);
  gsl_rng_free(gslrng);
  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
