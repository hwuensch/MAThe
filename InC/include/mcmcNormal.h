
int getKovMat(gsl_matrix* KovMat, int type, int dimension);
int getProposal(const gsl_rng* gslrng, int dimension, double* thetaCurr, double thetaCan[], double* qCurr, double* qCan);
int getPosterior(gsl_vector* thetaCurrV, int dimension, double* posteriorCurr);
int getStarted(double startvalue, int dimension, gsl_vector* thetaV, double* posterior);
int getAcceptancelevel(double* posteriorCan, double* posteriorCurr, double* qCan, double* qCurr, double* acceptlevel);
long getSeed();
