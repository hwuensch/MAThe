#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <mcmcNormal.h>

int main(int argc,char *argv[])
{
  int retval;
  int iterAll, iterJ, dimension;
  int world_rank, world_size;
  double starttime_setup;
  double startvalue, proposalType;
  double posteriorCurr, posteriorCan, qCurr, qCan;
  gsl_vector *thetaCanV, *thetaCurrV;
  double acceptlevel, acceptUniform;
  unsigned long seed, acceptrate=0;
  gsl_rng *gslrng;
  char filename_open[100];
  FILE *fileChain, *fileLog;

  // init MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  // starttime_setup = MPI_Wtime();

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
    startvalue = atof(argv[3]);
  } else {
    printf("No startvalue 'start' is defined. We work with start=2.0.\n");
    startvalue = 2.0;
  }
  // proposal
  if (argc>4) {
    proposalType = atof(argv[4]);
  } else {
    printf("No value for type of proposal 'prop' is defined. We work with prop=0.4.\n");
    proposalType = 0.4;
  }

  /* output file */
  sprintf(filename_open,"");
  sprintf(filename_open,"../output/iter%d_dim%d_start%g_prop%d_rank%d.txt",iterAll,dimension,startvalue,(int) (proposalType*100),world_rank);
  fileChain=fopen(filename_open,"w");
  if (fileChain==NULL) {
    perror("Failed open");
    MPI_Finalize();
    return(1);
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

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // loop
  for (iterJ = 0; iterJ < iterAll; iterJ++) {
    // for (int i = 0; i < dimension; i++) { fprintf(fileChain,"%.4e\t",gsl_vector_get(thetaCurrV,i)); }
    retval = writeToFile(fileChain, thetaCurrV);
    // Proposal
    retval = getProposal(gslrng, proposalType, dimension, thetaCurrV, thetaCanV, &qCurr, &qCan);
    for (int i = 0; i < dimension; i++) { fprintf(fileChain,"%.4e\t",gsl_vector_get(thetaCanV,i)); }

    // Posterior
    retval = getPosterior(thetaCanV, dimension, &posteriorCan);
    fprintf(fileChain,"%.6e\t%.6e\t%.6e\t%.6e\t", posteriorCan, qCurr, posteriorCurr, qCan);
    // Akzeptanzlevel
    retval =  getAcceptancelevel(&posteriorCan, &posteriorCurr, &qCan, &qCurr, &acceptlevel);
    // AccRej
    acceptUniform = gsl_rng_uniform(gslrng);
    fprintf(fileChain,"%.6e\t%.6e\t", acceptlevel, acceptUniform);
    if (acceptUniform < acceptlevel) {
      fprintf(fileChain,"1\t");
      acceptrate++;
      retval = gsl_vector_memcpy(thetaCurrV, thetaCanV);
      posteriorCurr = posteriorCan;
    } else {
      fprintf(fileChain,"0\t");
    }
    fprintf(fileChain,"\n");
  }
  for (int i = 0; i < dimension; i++) { fprintf(fileChain,"%.4e\t",gsl_vector_get(thetaCurrV,i)); }
  fprintf(fileChain,"\t\t\t\t\t\t\t\t\t%.4f",(double)acceptrate/iterAll);
  fprintf(fileChain,"\n");


  // free memory
  gsl_vector_free(thetaCanV);
  gsl_vector_free(thetaCurrV);
  gsl_rng_free(gslrng);
  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
