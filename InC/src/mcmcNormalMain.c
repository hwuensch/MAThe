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
  int iterAll, iterJ, dimension, swapBool, didSwap=0;
  int world_rank, world_size;
  double starttime_setup;
  double startvalue, proposalType;
  double posteriorCurr, posteriorCan, qCurr, qCan;
  gsl_vector *thetaCanV, *thetaCurrV, *thetaCurrV_recv;
  double acceptlevel, acceptUniform;
  unsigned long seed, acceptrate=0;
  gsl_rng *gslrng;
  char filename_open[100];
  FILE *fileChain;

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
  // bool swaps
  if (argc>5) {
    swapBool = atoi(argv[5]);
    if ((world_size < 3)&&(swapBool==1)) {
      // nach Paper kann/soll nur mit mindestens drei Ketten getauscht werden.
      printf("We need at least 3 chains to perform the swapping scheme.\n");
      swapBool = 0;
    }
  } else {
    printf("No value for swapping the chain 'swaps' is defined. We work without swaps; swaps=0.\n");
    swapBool = 0;
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
  thetaCurrV = gsl_vector_calloc(dimension); thetaCurrV_recv = gsl_vector_calloc(dimension);
  thetaCanV  = gsl_vector_calloc(dimension);
  retval = getStarted(startvalue, dimension, thetaCurrV, &posteriorCurr); // Startpunkt und dessen Werte setzen

  /****************************************************************************/
  /****************************************************************************/
  /****************************************************************************/
  // loop
  for (iterJ = 0; iterJ < iterAll; iterJ++) {
    retval = writeToFile(fileChain, thetaCurrV);
    if (swapBool) {
      // sPHS
      didSwap = performSwap(gslrng,thetaCurrV,&posteriorCurr);
    }
    if (didSwap) {
      for (int i = 0; i < dimension; i++) {fprintf(fileChain,"\t");} fprintf(fileChain,"\t\t\t\t\t\t2");
      acceptrate++;
    } else {
      // Proposal
      retval = getProposal(gslrng, proposalType, dimension, thetaCurrV, thetaCanV, &qCurr, &qCan);
      retval = writeToFile(fileChain, thetaCanV);
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
    }
    fprintf(fileChain,"\n");
  }
  retval = writeToFile(fileChain, thetaCurrV);
  for (int i = 0; i < dimension; i++) {fprintf(fileChain,"\t");}
  fprintf(fileChain,"\t\t\t\t\t\t%.4f",(double)acceptrate/iterAll);
  fprintf(fileChain,"\n");


  // free memory
  gsl_vector_free(thetaCanV);
  gsl_vector_free(thetaCurrV);
  gsl_vector_free(thetaCurrV_recv);
  gsl_rng_free(gslrng);
  // close files
  fclose(fileChain);
  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
