#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define PROP 1
#define POST 2

/******************************************************************************/

int getKovMat(double* KovMat, int type, int dimension){
  double inhalt;

  // Inhalt fuer KovMat festlegen
  switch (type) {
    case PROP:
      inhalt = pow(0.5,2);
      break;
    case POST:
      inhalt = pow(1.0,2);
      break;
    default: return(1); break;
  }
  // Kovarianzmatrix befuellen
  for (int i = 0; i < dimension; i++) {
    KovMat[i] = inhalt;
  }

  return(0);
}

/******************************************************************************/

int getProposal(int dimension, double* thetaCurr, double thetaCan[], double* qCan){
  // Proposal ist Random Walk um die aktuelle Position.
  // Die einzelnen Einträge sind unabhängig voneinander; können sie also einzeln würfeln.
  //
  // thetaCan = thetaCurr + sigma * ksi mit ksi = N(0,1).
  gsl_rng *r;
  int retval;
  double *KovMatProposal;

  KovMatProposal = (double *) calloc(dimension, sizeof(double));
  retval = getKovMat(KovMatProposal, PROP, dimension);

  if (*qCan == -1.0) {
    /* in diesem Fall moechte ich nur den Wert der Proposalverteilung */
    // Proposal ist Random Walk um aktuelle Position
    for (int i = 0; i < dimension; i++) {

    }
    return(0);
  }

  for (int i = 0; i < dimension; i++) {
    thetaCan[i] = i;
  }
  // for (int i = 0; i < dimension; i++) {
  //   thetaCan = thetaCurr + KovMatProposal[i] * gsl_ran_gaussian(gslrng,1.0);
  // }

  free(KovMatProposal);
  return(0);
}

/******************************************************************************/

int getPosterior(double* theta, int dimension, double* posterior){
  int retval;
  double nenner;
  double *MuPosterior, *KovMatPosterior;

  MuPosterior     = (double *) calloc(dimension, sizeof(double)); // Mittelwertvektor von Zielposterior ist Null
  KovMatPosterior = (double *) calloc(dimension, sizeof(double));
  retval = getKovMat(KovMatPosterior, POST, dimension);

  // mehrdimensionale Normalverteilung:
  // Kovarianzmatrix ist diagonal -> det(.) ist das Produkt der einzelnen Einträge
  *posterior = 0.0; nenner = 1.0;
  for (int i = 0; i < dimension; i++) {
    nenner *= 2*M_PI * KovMatPosterior[i];
    *posterior += pow(theta[i] - MuPosterior[i],2) / KovMatPosterior[i];
  }
  *posterior = exp(-0.5 * *posterior) / sqrt(nenner);

  free(MuPosterior);
  free(KovMatPosterior);
  return(0);
}

/******************************************************************************/

int getStarted(int dimension, double* theta, double* posterior, double* proposal){
  int retval;
  double startValue = 1.0;

  for (int i = 0; i < dimension; i++) {
    theta[i] = startValue;
  }

  retval = getPosterior(theta, dimension, posterior);

  *proposal = -1.0;
  retval = getProposal(dimension, theta, theta, proposal);

  return(0);
}
