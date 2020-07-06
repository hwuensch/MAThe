
int getKovMat(double* KovMat, int type, int dimension);
int getProposal(int dimension, double* thetaCurr, double thetaCan[], double* qCurr, double* qCan);
int getPosterior(double* thetaCurr, int dimension, double* posteriorCurr);
int getStarted(int dimension, double* thetaCurr, double* posteriorCurr);
