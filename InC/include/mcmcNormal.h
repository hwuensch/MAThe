
int getKovMat(double* KovMat, int type, int dimension);
int getProposal(const gsl_rng* gslrng, int dimension, double* thetaCurr, double thetaCan[], double* qCurr, double* qCan);
int getPosterior(double* thetaCurr, int dimension, double* posteriorCurr);
int getStarted(double startvalue, int dimension, double* thetaCurr, double* posteriorCurr);
int getAcceptancelevel(double* posteriorCan, double* posteriorCurr, double* qCan, double* qCurr, double* acceptlevel);
long getSeed();
