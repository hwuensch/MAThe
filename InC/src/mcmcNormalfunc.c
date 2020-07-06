#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define PROP 1
#define POST 2

int getProposal(double* KovMatProposal, int dimension, double* thetaCurr, double thetaCan[], double* qCan){
  // Proposal ist Random Walk um die aktuelle Position.
  // Die einzelnen Einträge sind unabhängig voneinander; können sie also einzeln würfeln.
  //
  // thetaCan = thetaCurr + sigma * ksi mit ksi = N(0,1).
  gsl_rng *r;

  for (int i = 0; i < dimension; i++) {
    thetaCan[i] = i;
  }
  // for (int i = 0; i < dimension; i++) {
  //   thetaCan = thetaCurr + KovMatProposal[i] * gsl_ran_gaussian(gslrng,1.0);
  // }
  return(0);
}

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
