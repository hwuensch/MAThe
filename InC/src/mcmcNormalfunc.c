#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void getProposal(double* KovMatProposal, int dim, double* thetaCurr, double thetaCan[], double* qCan){
  // Proposal ist Random Walk um die aktuelle Position.
  // Die einzelnen Einträge sind unabhängig voneinander; können sie also einzeln würfeln.
  //
  // thetaCan = thetaCurr + sigma * ksi mit ksi = N(0,1).
  int dimension = dim;
  gsl_rng *r;

  for (int i = 0; i < dimension; i++) {
    thetaCan[i] = i;
  }
  // for (int i = 0; i < dimension; i++) {
  //   thetaCan = thetaCurr + KovMatProposal[i] * gsl_ran_gaussian(gslrng,1.0);
  // }
}
