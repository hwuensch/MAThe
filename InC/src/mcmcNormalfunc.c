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
#define POSTZURL 3.0

/******************************************************************************/

int getKovMat(gsl_matrix* KovMat, double type, int dimension){
  double inhalt;

  // Inhalt fuer KovMat festlegen
  if (type == POST) {
    inhalt = 1.0;
  } else {
    inhalt = pow(type,2);
  }

  // Kovarianzmatrix befuellen
  for (int i = 0; i < dimension; i++) {
    gsl_matrix_set(KovMat, i, i, inhalt);
  }

  if (type == POSTZURL) {
    gsl_matrix_set(KovMat,0,0,0.1219);
    gsl_matrix_set(KovMat,1,1,0.0862);    gsl_matrix_set(KovMat,2,2,0.0808);
    gsl_matrix_set(KovMat,3,3,0.5636);    gsl_matrix_set(KovMat,4,4,0.1416);
    gsl_matrix_set(KovMat,5,5,0.0606);    gsl_matrix_set(KovMat,6,6,0.1484);
    gsl_matrix_set(KovMat,7,7,0.1349);    gsl_matrix_set(KovMat,8,8,0.1034);
    gsl_matrix_set(KovMat,9,9,0.0654);    gsl_matrix_set(KovMat,10,10,0.1349);
    gsl_matrix_set(KovMat,11,11,0.0918); gsl_matrix_set(KovMat,12,12,0.0704);
    gsl_matrix_set(KovMat,13,13,0.1844); gsl_matrix_set(KovMat,14,14,0.0392);
    gsl_matrix_set(KovMat,15,15,1.3144); gsl_matrix_set(KovMat,16,16,0.5636);
    gsl_matrix_set(KovMat,17,17,0.3894); gsl_matrix_set(KovMat,18,18,0.1156);
    gsl_matrix_set(KovMat,19,19,0.0473);
    // Parameter 21: 0.0755
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
  double scale;
  gsl_vector *workspace;
  gsl_matrix *KovMatProposalCholesky, *Id, *KovMatPosterior;

  workspace = gsl_vector_calloc(dimension);
  KovMatProposalCholesky = gsl_matrix_calloc(dimension,dimension);
  Id = gsl_matrix_alloc(dimension,dimension);
  gsl_matrix_set_identity(Id);
  KovMatPosterior = gsl_matrix_calloc(dimension,dimension);

  retval = getKovMat(KovMatProposalCholesky, proposalType, dimension);
  // retval = getKovMat(KovMatPosterior, POSTZURL, dimension);
  // scale = 0.02;
  // retval = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, scale, KovMatPosterior, Id, 0.0, KovMatProposalCholesky);
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
  gsl_matrix_free(Id);
  gsl_matrix_free(KovMatPosterior);
  return(0);
}

/******************************************************************************/

int getPosterior(gsl_vector* thetaV, int dimension, double* posteriorX){
  int retval;
  gsl_vector *MuPosterior, *workspace;
  gsl_matrix *KovMatPosterior;

  workspace       = gsl_vector_calloc(dimension); // fuer gaussian_pdf notwendig
  MuPosterior     = gsl_vector_calloc(dimension); // Mittelwertvektor von Zielposterior ist Null
  if (dimension == 20) {
    int i = 0;
    gsl_vector_set(MuPosterior,i++,-1.1636); gsl_vector_set(MuPosterior,i++,-3.0880);
    gsl_vector_set(MuPosterior,i++,4.7718); gsl_vector_set(MuPosterior,i++,0.6621);
    gsl_vector_set(MuPosterior,i++,7.0193); gsl_vector_set(MuPosterior,i++,6.1393);
    gsl_vector_set(MuPosterior,i++,-1.1384); gsl_vector_set(MuPosterior,i++,6.0789);
    gsl_vector_set(MuPosterior,i++,8.6709); gsl_vector_set(MuPosterior,i++,10.7851);
    gsl_vector_set(MuPosterior,i++,-1.1375); gsl_vector_set(MuPosterior,i++,8.5125);
    gsl_vector_set(MuPosterior,i++,9.7348); gsl_vector_set(MuPosterior,i++,10.3823);
    gsl_vector_set(MuPosterior,i++,9.9927); gsl_vector_set(MuPosterior,i++,15.7974);
    gsl_vector_set(MuPosterior,i++,10.2095); gsl_vector_set(MuPosterior,i++,8.0105);
    gsl_vector_set(MuPosterior,i++,-4.4559); gsl_vector_set(MuPosterior,i++,-1.8880);
  }
  // Parameter 21: -2.0182
  KovMatPosterior = gsl_matrix_calloc(dimension,dimension);

  retval = getKovMat(KovMatPosterior, POSTZURL, dimension);

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

  if (dimension == 20) {
    int i = 0;
    gsl_vector_set(thetaV,i++,-1.1636); gsl_vector_set(thetaV,i++,-3.0880);
    gsl_vector_set(thetaV,i++,4.7718);  gsl_vector_set(thetaV,i++,0.6621);
    gsl_vector_set(thetaV,i++,7.0193);  gsl_vector_set(thetaV,i++,6.1393);
    gsl_vector_set(thetaV,i++,-1.1384); gsl_vector_set(thetaV,i++,6.0789);
    gsl_vector_set(thetaV,i++,8.6709);  gsl_vector_set(thetaV,i++,10.7851);
    gsl_vector_set(thetaV,i++,-1.1375); gsl_vector_set(thetaV,i++,8.5125);
    gsl_vector_set(thetaV,i++,9.7348);  gsl_vector_set(thetaV,i++,10.3823);
    gsl_vector_set(thetaV,i++,9.9927);  gsl_vector_set(thetaV,i++,15.7974);
    gsl_vector_set(thetaV,i++,10.2095); gsl_vector_set(thetaV,i++,8.0105);
    gsl_vector_set(thetaV,i++,-4.4559); gsl_vector_set(thetaV,i++,-1.8880);
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

int performSwap(FILE* file, const gsl_rng* gslrng, gsl_vector* thetaCurrV, double* posteriorCurr){
  // symmetrized Parallel Hierarchical Sampling:
  // genau zwei Ketten tauschen ihre aktuelle Position in jeder Iteration,
  // anstatt zu samplen.
  int world_rank, world_size, swap[2], dimension, tag=42, retval=0;
  double starttime;
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
  starttime = MPI_Wtime();
  MPI_Bcast(swap,2,MPI_INT,0,MPI_COMM_WORLD);
  starttime = MPI_Wtime() - starttime;
  fprintf(file,"%.4e\n",starttime);

  starttime = 0.0;
  if (swap[0]==world_rank) {
    starttime = MPI_Wtime();
    MPI_Sendrecv_replace(thetaCurrV->data,dimension,MPI_DOUBLE,swap[1],tag,swap[1],tag,MPI_COMM_WORLD,&status);
    MPI_Sendrecv_replace(posteriorCurr,1,MPI_DOUBLE,swap[1],tag,swap[1],tag,MPI_COMM_WORLD,&status);
    retval = 1;
    starttime = MPI_Wtime() - starttime;
  } else if (swap[1]==world_rank) {
    starttime = MPI_Wtime();
    MPI_Sendrecv_replace(thetaCurrV->data,dimension,MPI_DOUBLE,swap[0],tag,swap[0],tag,MPI_COMM_WORLD,&status);
    MPI_Sendrecv_replace(posteriorCurr,1,MPI_DOUBLE,swap[0],tag,swap[0],tag,MPI_COMM_WORLD,&status);
    retval = 1;
    starttime = MPI_Wtime() - starttime;
  }
  fprintf(file,"%.4e\n",starttime);

  return retval;
}

/******************************************************************************/
