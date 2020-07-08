#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
// #include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>

#define PROP 1
#define POST 2

/******************************************************************************/

int getKovMat(gsl_matrix* KovMat, int type, int dimension){
  double inhalt;

  // Inhalt fuer KovMat festlegen
  switch (type) {
    case PROP:
      inhalt = pow(0.4,2);
      break;
    case POST:
      inhalt = pow(1.0,2);
      break;
    default: return(1); break;
  }
  // Kovarianzmatrix befuellen
  for (int i = 0; i < dimension; i++) {
    gsl_matrix_set(KovMat, i, i, inhalt);
  }

  return(0);
}

/******************************************************************************/

int getProposal(const gsl_rng* gslrng, int dimension, gsl_vector* thetaCurr, gsl_vector* thetaCan, double* qCurr, double* qCan){
  // Proposal ist Random Walk um die aktuelle Position.
  // Die einzelnen Einträge sind unabhängig voneinander; können sie also einzeln würfeln.
  //
  // thetaCan = thetaCurr + sigma * ksi mit ksi = N(0,1).
  // gsl_rng *gslrng;
  int retval;
  // double nennerCurr, nennerCan;
  // double *KovMatProposal;
  //
  // gsl_vector *thetaCanV, *thetaCurrV;
  // gsl_matrix *KovMatProposalCholesky;
  //
  // thetaCanV  = gsl_vector_calloc(dimension);
  // thetaCurrV = gsl_vector_calloc(dimension);
  // for (int i = 0; i < dimension; i++) {
  //   gsl_vector_set(thetaCurrV, i, thetaCurr[i]);
  // }
  //
  // KovMatProposal = (double *) calloc(dimension, sizeof(double));
  // retval = getKovMat(KovMatProposal, PROP, dimension);
  // KovMatProposalCholesky = gsl_matrix_calloc(dimension,dimension);
  // for (int i = 0; i < dimension; i++) {
  //   gsl_matrix_set(KovMatProposalCholesky, i, i, KovMatProposal[i]);
  // }
  //
  // // Kandidaten wuerfeln:
  // /* via gsl:
  //  * int gsl_linalg_cholesky_decomp1(gsl_matrix * A) -> Error GSL_EDOM, falls nicht positiv definit
  //  * int gsl_ran_multivariate_gaussian(const gsl_rng * r, const gsl_vector * mu, const gsl_matrix * L, gsl_vector * result)
  //  */
  // retval = gsl_linalg_cholesky_decomp1(KovMatProposalCholesky);
  // if (retval == GSL_EDOM) { printf("KovMat der Proposal ist nicht spd!\n");}
  // retval = gsl_ran_multivariate_gaussian(gslrng, thetaCurrV, KovMatProposalCholesky, thetaCanV);
  //
  // // Da ich einen Random Walk ohne Kovarianzen mache, kann ich jeden Eintrag
  // // einzeln wuerfeln.
  // for (int i = 0; i < dimension; i++) {
  //   thetaCan[i] = thetaCurr[i] + sqrt(KovMatProposal[i]) * gsl_ran_gaussian(gslrng,1.0);
  // }
  //
  // // Wahrscheinlichkeiten qCurr und qCan berechnen:
  // // Kovarianzmatrix ist diagonal -> det(.) ist das Produkt der einzelnen Einträge
  // nennerCan = 1.0; nennerCurr = 1.0; *qCan = 0.0; *qCurr = 0.0;
  // for (int i = 0; i < dimension; i++) {
  //   nennerCan  *= 2*M_PI * KovMatProposal[i];
  //   nennerCurr = nennerCan;
  //   *qCan      += pow(thetaCan[i] - thetaCurr[i],2) / KovMatProposal[i];
  //   *qCurr     += pow(thetaCurr[i] - thetaCan[i],2) / KovMatProposal[i];
  // }
  // *qCan  = exp(-0.5 * *qCan) / sqrt(nennerCan);
  // *qCurr = exp(-0.5 * *qCurr) / sqrt(nennerCurr);
  //
  // free(KovMatProposal);
  // gsl_vector_free(thetaCanV);
  // gsl_vector_free(thetaCurrV);
  // gsl_matrix_free(KovMatProposalCholesky);
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
  int retval;

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
