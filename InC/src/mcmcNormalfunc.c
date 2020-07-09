#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
// #include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>

#define PROP 1.0
#define POST 2.0

/******************************************************************************/

int getKovMat(gsl_matrix* KovMat, double type, int dimension){
  double inhalt;

  // Inhalt fuer KovMat festlegen
  if (type == POST) {
    inhalt = pow(1.0,2);
  } else {
    inhalt = pow(type,2);
  }

  // Kovarianzmatrix befuellen
  for (int i = 0; i < dimension; i++) {
    gsl_matrix_set(KovMat, i, i, inhalt);
  }

  return(0);
}

/******************************************************************************/

int getProposal(const gsl_rng* gslrng, double proposalType, int dimension, gsl_vector* thetaCurrV, gsl_vector* thetaCanV, double* qCurr, double* qCan){
  // Proposal ist Random Walk um die aktuelle Position.
  // Die einzelnen Einträge sind unabhängig voneinander; können sie also einzeln würfeln.
  //
  // thetaCan = thetaCurr + sigma * ksi mit ksi = N(0,1).

  int retval;
  gsl_vector *workspace;
  gsl_matrix *KovMatProposalCholesky;

  workspace = gsl_vector_calloc(dimension);
  KovMatProposalCholesky = gsl_matrix_calloc(dimension,dimension);
  retval = getKovMat(KovMatProposalCholesky, proposalType, dimension);

  // Kandidaten wuerfeln:
  /* via gsl:
   * int gsl_linalg_cholesky_decomp1(gsl_matrix * A) -> Error GSL_EDOM, falls nicht positiv definit
   * int gsl_ran_multivariate_gaussian(const gsl_rng * r, const gsl_vector * mu, const gsl_matrix * L, gsl_vector * result)
   */
  retval = gsl_linalg_cholesky_decomp1(KovMatProposalCholesky);
  if (retval == GSL_EDOM) { printf("KovMat der Proposal ist nicht spd!\n");}
  retval = gsl_ran_multivariate_gaussian(gslrng, thetaCurrV, KovMatProposalCholesky, thetaCanV);
  // Wahrscheinlichkeiten qCurr und qCan berechnen: (bei symmetrischer Proposal gleich)
  retval = gsl_ran_multivariate_gaussian_pdf(thetaCanV, thetaCurrV, KovMatProposalCholesky, qCan, workspace);
  retval = gsl_ran_multivariate_gaussian_pdf(thetaCurrV, thetaCanV, KovMatProposalCholesky, qCurr, workspace);

  gsl_vector_free(workspace);
  gsl_matrix_free(KovMatProposalCholesky);
  return(0);
}

/******************************************************************************/

int getPosterior(gsl_vector* thetaV, int dimension, double* posteriorX){
  int retval;
  gsl_vector *MuPosterior, *workspace;
  gsl_matrix *KovMatPosterior;

  workspace       = gsl_vector_calloc(dimension); // fuer gaussian_pdf notwendig
  MuPosterior     = gsl_vector_calloc(dimension); // Mittelwertvektor von Zielposterior ist Null
  KovMatPosterior = gsl_matrix_calloc(dimension,dimension);
  retval = getKovMat(KovMatPosterior, POST, dimension);

  // mehrdimensionale Normalverteilung:
  retval = gsl_linalg_cholesky_decomp1(KovMatPosterior);
  retval = gsl_ran_multivariate_gaussian_pdf(thetaV, MuPosterior, KovMatPosterior, posteriorX, workspace);

  gsl_vector_free(MuPosterior);
  gsl_vector_free(workspace);
  gsl_matrix_free(KovMatPosterior);
  return(0);
}

/******************************************************************************/

int getStarted(double startvalue, int dimension, gsl_vector* thetaV, double* posterior){
  int retval;

  for (int i = 0; i < dimension; i++) {
    gsl_vector_set(thetaV, i, startvalue);
  }

  retval = getPosterior(thetaV, dimension, posterior);

  return(0);
}

/******************************************************************************/

int getAcceptancelevel(double* posteriorCan, double* posteriorCurr, double* qCan, double* qCurr, double* acceptlevel){

  *acceptlevel = *posteriorCan * *qCan / *posteriorCurr / *qCurr;

  return(0);
}

/******************************************************************************/

long getSeed(){
    /* calculates and returns the current time
     */
    struct timeval timevalue;

    gettimeofday(&timevalue, 0);
    return(timevalue.tv_sec + timevalue.tv_usec);
}

/******************************************************************************/

int writeToFile(FILE* file, const gsl_vector* vector){
  int dimension;

  dimension = vector->size;

  for (int i = 0; i < dimension; i++){
    fprintf(file,"%.4e\t",gsl_vector_get(vector,i));
  }

  return(0);
}

/******************************************************************************/

int performSwap(const gsl_rng* gslrng, gsl_vector* thetaCurrV, double* posteriorCurr){
  // symmetrized Parallel Hierarchical Sampling:
  // genau zwei Ketten tauschen ihre aktuelle Position in jeder Iteration,
  // anstatt zu samplen.
  int world_rank, world_size, swap[2], dimension, tag=42, retval=0;
  MPI_Status status;

  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  dimension = thetaCurrV->size;

  if (world_rank==0) {
    swap[0] = gsl_rng_uniform_int(gslrng, world_size);
    swap[1] = gsl_rng_uniform_int(gslrng, world_size-1);
    if (swap[1] >= swap[0]) {
      // swap[1] wurde aus einer Zahl weniger gezogen -> korrekte Position muss evtl angepasst werden.
      swap[1]++;
    }
  }
  MPI_Bcast(swap,2,MPI_INT,0,MPI_COMM_WORLD);

  if (swap[0]==world_rank) {
    MPI_Sendrecv_replace(thetaCurrV->data,dimension,MPI_DOUBLE,swap[1],tag,swap[1],tag,MPI_COMM_WORLD,&status);
    MPI_Sendrecv_replace(posteriorCurr,1,MPI_DOUBLE,swap[1],tag,swap[1],tag,MPI_COMM_WORLD,&status);
    retval = 1;
  } else if (swap[1]==world_rank) {
    MPI_Sendrecv_replace(thetaCurrV->data,dimension,MPI_DOUBLE,swap[0],tag,swap[0],tag,MPI_COMM_WORLD,&status);
    MPI_Sendrecv_replace(posteriorCurr,1,MPI_DOUBLE,swap[0],tag,swap[0],tag,MPI_COMM_WORLD,&status);
    retval = 1;
  }
  return retval;
}

/******************************************************************************/
