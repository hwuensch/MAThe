#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <mcmcNormal.h>



int main(int argc,char *argv[])
{
  int iterAll, iterJ, dim;
  int world_rank, world_size;
  double starttime_setup;
  double *thetaCan, *thetaCurr;
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
    return(1);
  }

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // Startpunkt
  thetaCan  = (double *) malloc(dim * sizeof(double));
  thetaCurr = (double *) malloc(dim * sizeof(double));
  // getStarted(dim, thetaCurr, posteriorCurr, qCurr); // Startpunkt und dessen Werte setzen
  // // Kovarianzmatrix Proposalverteilung
  // getKovMat(KovMatProposal);
  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // loop
  for (int iterJ = 0; iterJ < iterAll; iterJ++) {
    // // Proposal
    // getProposal(KovMatProposal, dim, thetaCurr, thetaCan, &qCan);
    // // Posterior
    // getPosterior(thetaCan, posteriorCan);
    // // Akzeptanzlevel
    // acceptlevel =  getAcceptancelevel(posteriorCan, posteriorCurr, qCan, qCurr);
    // // AccRej
    // acceptUniform = gsl_rng_uniform(gslrng);
    // if (acceptUniform < acceptlevel) {
    //   thetaCurr = thetaCan;
    //   posteriorCurr = posteriorCan;
    // } else {
    //
    // }
  }

  // free memory
  free(thetaCan);
  free(thetaCurr);
  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
