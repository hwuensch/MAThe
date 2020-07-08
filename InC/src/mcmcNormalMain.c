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
  int iterAll, iterJ, dimension;
  int world_rank, world_size;
  double starttime_setup;
  double startvalue;
  double posteriorCurr, posteriorCan, qCurr, qCan;
  gsl_vector *thetaCanV, *thetaCurrV;
  double acceptlevel, acceptUniform;
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
    dimension = atoi(argv[2]);
  } else {
    printf("No dimension 'dim' is defined. We work with dim=20.\n");
    dimension = 20;
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

  // Startpunkt
  thetaCurrV = gsl_vector_calloc(dimension);
  thetaCanV  = gsl_vector_calloc(dimension);
  retval = getStarted(startvalue, dimension, thetaCurrV, &posteriorCurr); // Startpunkt und dessen Werte setzen
  printf("x_0 = (");for (size_t i = 0; i < dimension; i++) {printf("%g\t",gsl_vector_get(thetaCurrV,i));} printf(")\n");
  printf("p(x_0) = %g\n",posteriorCurr);

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // loop
  for (int iterJ = 0; iterJ < iterAll; iterJ++) {
    // Proposal
    retval = getProposal(gslrng, dimension, thetaCurrV, thetaCanV, &qCurr, &qCan);
    printf("candid_%3d\t",iterJ); for (int i = 0; i < dimension; i++) { printf("%.2f\t",gsl_vector_get(thetaCanV,i)); }

    // Posterior
    retval = getPosterior(thetaCanV, dimension, &posteriorCan);
    printf("%.2e\t%.3f\t%.2e\t%.3f\t", posteriorCan, qCurr, posteriorCurr, qCan);
    // Akzeptanzlevel
    retval =  getAcceptancelevel(&posteriorCan, &posteriorCurr, &qCan, &qCurr, &acceptlevel);
    // AccRej
    acceptUniform = gsl_rng_uniform(gslrng);
    printf("%.3f\t>\t%.3f\t", acceptlevel, acceptUniform);
    if (acceptUniform < acceptlevel) {
      printf("YES\t");
      retval = gsl_vector_memcpy(thetaCurrV, thetaCanV);
      posteriorCurr = posteriorCan;
    } else {
      printf("no\t");
    }
    printf("curr_%3d\t",iterJ); for (int i = 0; i < dimension; i++) { printf("%.2f\t",gsl_vector_get(thetaCurrV,i)); }
    printf("\n");
  }

  // free memory
  gsl_vector_free(thetaCanV);
  gsl_vector_free(thetaCurrV);
  gsl_rng_free(gslrng);
  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
