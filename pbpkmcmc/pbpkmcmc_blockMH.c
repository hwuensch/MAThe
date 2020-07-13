/* -----------------------------------------------------------------
 * blockMH:
 *
 * Der blockweise Metropolis-Hastings mit symmetrischem Random-Walk
 * für jeden Block als Proposalverteilung und je nach Übergabeparameter
 * mehrkettig und/oder mit Positionstausch.
 * -----------------------------------------------------------------
 * Bemerkungen bzgl. Sundials:
 *
 * The problem is stiff.
 * This program solves the problem with the BDF method,
 * Newton iteration with the SUNDENSE dense linear solver.
 * It uses a scalar relative tolerance and a vector absolute
 * tolerance.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl-sprng.h>
#include <gsl/gsl_randist.h>
#include <mpi.h>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */

#include <pbpkmcmc.h>

/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ] and NEQ is defined below. The Ith macro is defined
   using the N_VIth macro in nvector.h. N_VIth numbers the components of
   a vector starting from 0.

   IJth(A,i,j) references the (i,j)th element of the dense matrix A, where
   i and j are in the range [1..NEQ]. The IJth macro is defined using the
   SM_ELEMENT_D macro in dense.h. SM_ELEMENT_D numbers rows and columns of
   a dense matrix starting from 0. */

#define Ith(v,i)    NV_Ith_S(v,i-1)         /* Ith numbers components 1..NEQ */
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* IJth numbers rows,cols 1..NEQ */

#define ZERO  RCONST(0.0)

/* Problem Constants */
#define NSTUDIES 10
#define NPARALL  211
#define NPARSTUDY 21
#define NEQ   32               /* number of equations  */
//#define AMT   RCONST(5.0e+3)   /* amount of drug dosis */
#define RTOL  RCONST(1.0e-6)   /* scalar relative tolerance            */
#define ATOL  RCONST(1.0e-14)  /* vector absolute tolerance components */
#define T0    RCONST(0.0)      /* initial time           */
//#define T1    RCONST(0.016666666666667)  /* first output time (1min = 0.0166...)     */
//#define DTOUT RCONST(0.016666666666667)
//#define NOUT  720              /* number of output times (1h: NOUT=60, 12h: NOUT=720) */
#define MXSTEPS 50000            /* max number of steps to reach next output time */
// for test run:
#define BODYWEIGHT  RCONST(50.0)
#define DOSISORAL   RCONST(20.0)
#define DOSISREL    RCONST(1.0) // 1 = relative Dosis, -1 = absolute Dosis

// MPI Konstanten
#define RANKMASTER 0

/* Type : UserData (contains some variables) */
typedef struct {
  realtype BW, D_oral, D_oral_rel;
  realtype T_G, T_P, K_APAP_Mcyp, V_MCcyp, K_APAP_Msult, K_APAP_Isult;
  realtype K_PAPS_Msult, V_MCsult, K_APAP_Mugt, K_APAP_Iugt, K_UDPGA_Mugt;
  realtype V_MCugt, K_APAPG_Mmem, V_APAPG_Mmem, K_APAPS_Mmem, V_APAPS_Mmem;
  realtype k_synUDPGA, k_synPAPS, k_APAP_R0, k_APAPG_R0, k_APAPS_R0;
} *UserData;

/******************************************************************************/

/******************************************************************************
 *-------------------------------
 * Main Program
 *-------------------------------
 ******************************************************************************/

int main(int argc,char *argv[])
{
  double starttime_iterAll, endtime_iterAll;
  double starttime_proposal, endtime_proposal, endtime_posterior, endtime_accrej;
  double starttime_ODE, endtime_getInfos, endtime_ODE_solve, endtime_ODE_solve_fail, endtime_ODE_IV, endtime_ODE_reinit, endtime_posterior_rest;
  double endtime_ODE, endtime_ODE_fail;
  double endtime_swap_toss, endtime_swap_bcast, endtime_swap_barrier, endtime_swap_partner, endtime_swap_wait, endtime_swap;
  double starttime_sigmaLL, endtime_sigmaLL;
  double cputime_ODE_sum=0.0, cputime_iterAll, cputime_swap=0.0;
  double cputime_proposal=0.0, cputime_posterior=0.0, cputime_accrej=0.0;
  double cputime_getInfos=0.0, cputime_ODE_solve=0.0, cputime_ODE_IV=0.0, cputime_ODE_reinit=0.0, cputime_ODE_fail=0.0, cputime_posterior_rest=0.0;
  double cputime_swap_toss=0.0, cputime_swap_bcast=0.0, cputime_swap_barrier=0.0, cputime_swap_partner=0.0, cputime_swap_wait=0.0;
  double thetaCurr_swap_send[NPARALL], thetaCurr_swap_recv[NPARALL], logPosterior_swap_send[NSTUDIES+1], logPosterior_swap_recv[NSTUDIES+1];
  realtype reltol, t, tout;
  N_Vector y_Sundials, abstol, thetaCurr, thetaCan, thetaCan_temp, odeSolution, posteriorZurlindenMean, posteriorZurlindenCoV;
  N_Vector data2fit, logPosterior_can, logPosterior_curr, LikelihoodNumerator, nAccepted;
  UserData data;
  SUNMatrix A;
  SUNLinearSolver LS;
  void *cvode_mem;
  int world_rank, world_size, tag=42, swapPartner;
  int retval, retval2, iout, i, ctr, iterJ, k, kk, iterAll, nData, nAPAP_G_S, studyblock, colsODESolu=3, LogNormalBool=0;
  int rootsfound[2], logLikelihood_nan, ode_fail=0, waitbar, waitbar_counter, onepercent, nAcceptBool, nAcceptBool_print;
  int swaps_max, swaps, seedBool;
  long nIterSundials=0, nIterSundials_sum=0, nIterSundials_fail=0;
  double x,can,logaccrate,logaccrateUniform, acceptrate;
  realtype mu, sigmaRW, sigmaRW_scale, max, min, thetaCanI;
  realtype logPrior_can, logPrior_can_temp, logLikelihood_can, logLikelihood_can_blockj, logPosterior, LikelihoodNumerator_blockj, logPosterior_swap;
  gsl_rng *gslrng;
  FILE *fileLog, *fileSolution, *fileMarkovChain, *fileAcceptReject, *fileCandidates, *fileTimes, *filenIterSundials;
  char filename[40], filename_dir[50], filename_open[100];
  unsigned long seed;
  MPI_Status status, status_swap_theta, status_swap_logPosterior;
  MPI_Request request_swap_logPosterior, request_swap_theta;

  realtype zufallszahl;

  // init MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  /* get inputs */
  // Constants for MCMC loop
  if (argc>1) {
    iterAll = atoi(argv[1]);
  } else {
      MPI_Finalize();
      printf("A constant (int) for # of iterations is needed.\nEnd.\n");
      return(1);
  }
  // Constants for swap
  swaps_max=world_size;
  if (argc>2) {
    swaps = atoi(argv[2]);
  } else {
    swaps = 0;
    printf("No input for # of swapping chains is given -> we work without swaps.\n");
  }
  if (swaps>swaps_max) {
    swaps = swaps_max;
    printf("We were asked for more chains to swap than possible --> set it onto max. possible: %d\n", swaps);
  }
  if ((swaps_max>1)&&(swaps%2==1)) {
    printf("# for swaps is uneven: %d", swaps);
    swaps--;
    printf(" --> we work with %d\n", swaps);
  }
  // Constants for seed
  if (argc>3) {
    seedBool = atoi(argv[3]);
  } else {
    seedBool = 0;
    printf("No input for boolean seed. Take default seed.\n");
  }

  // create log-files
  sprintf(filename_dir,""); sprintf(filename_open,""); sprintf(filename_dir,"");

  sprintf(filename_dir,"output/blockMH/%d/%d_%d_",iterAll,world_size,swaps);
  strcat(filename_open,filename_dir);

  sprintf(filename,"log_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileLog=fopen(filename_open,"a");
  if (fileLog==NULL) {
    fileLog = fopen(filename,"w");
  }
  if (fileLog==NULL) {
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  // write initial infos into log-file
  fprintf(fileLog,"We have %d ranks initialized.\n", world_size);
  fprintf(fileLog,"We have %d iterations initialized.\n", iterAll);
  fprintf(fileLog,"We perform the swap between %d chains.\n", swaps);

  // preallocate and create vector for parameters
  thetaCurr = NULL; thetaCurr = N_VNew_Serial(NPARALL); if (check_retval((void *)thetaCurr, "N_VNew_Serial", 0)) return(1);
  thetaCan = NULL; thetaCan = N_VNew_Serial(NPARALL); if (check_retval((void *)thetaCan, "N_VNew_Serial", 0)) return(1);
  thetaCan_temp = NULL; thetaCan_temp = N_VNew_Serial(NPARSTUDY); if (check_retval((void *)thetaCan_temp, "N_VNew_Serial", 0)) return(1);
  posteriorZurlindenMean = NULL; posteriorZurlindenMean = N_VNew_Serial(NPARSTUDY); if (check_retval((void *)posteriorZurlindenMean, "N_VNew_Serial", 0)) return(1);
  posteriorZurlindenCoV = NULL; posteriorZurlindenCoV = N_VNew_Serial(NPARSTUDY); if (check_retval((void *)posteriorZurlindenCoV, "N_VNew_Serial", 0)) return(1);

  logPosterior_can = NULL; logPosterior_can = N_VNew_Serial(NSTUDIES+1); if (check_retval((void *)logPosterior_can, "N_VNew_Serial", 0)) return(1);
  logPosterior_curr = NULL; logPosterior_curr = N_VNew_Serial(NSTUDIES+1); if (check_retval((void *)logPosterior_curr, "N_VNew_Serial", 0)) return(1);
  LikelihoodNumerator = NULL; LikelihoodNumerator = N_VNew_Serial(NSTUDIES); if (check_retval((void *)LikelihoodNumerator, "N_VNew_Serial", 0)) return(1);

  nAccepted = NULL; nAccepted = N_VNew_Serial(NSTUDIES+1); if(check_retval((void *)nAccepted, "N_VNew_Serial", 0)) return(1);
  // preallocate vector to save the ode odeSolution
  odeSolution = NULL;

  // preallocate variables for sundials solver
  y_Sundials = abstol = NULL;
  A = NULL;
  LS = NULL;
  cvode_mem = NULL;

  // create an instance of a random number generator of type gsl_rng_sprng20
  // gslrng = gsl_rng_alloc(gsl_rng_sprng20);
  gslrng = gsl_rng_alloc(gsl_rng_taus);
  if (seedBool!=0) {
    seed    = getSeed();            // get a seed based on current time. default seed = 0.
    gsl_rng_set(gslrng, seed);      // set a different seed for the rng.
  }

  /* create output files */
  sprintf(filename,"solu_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileSolution=fopen(filename_open,"w");
  if (fileSolution==NULL) {
      perror("Failed open");
      MPI_Finalize();
      return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  sprintf(filename,"AccRej_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileAcceptReject=fopen(filename_open,"w");
  if(fileAcceptReject==NULL){
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  sprintf(filename,"chain_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileMarkovChain=fopen(filename_open,"w");
  if(fileMarkovChain==NULL){
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  sprintf(filename,"candidate_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileCandidates=fopen(filename_open,"w");
  if(fileCandidates==NULL){
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  sprintf(filename,"times_%d.txt",world_rank);
  strcat(filename_open,filename);
  fileTimes=fopen(filename_open,"a"); // append to file
  if (fileTimes==NULL) {
    fileTimes = fopen(filename,"w");
  }
  if (fileTimes==NULL) {
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  sprintf(filename,"nIterSundials_%d.txt",world_rank);
  strcat(filename_open,filename);
  filenIterSundials=fopen(filename_open,"a"); // append to file
  if (filenIterSundials==NULL) {
    filenIterSundials = fopen(filename,"w");
  }
  if (filenIterSundials==NULL) {
    perror("Failed open");
    MPI_Finalize();
    return(1);
  }
  sprintf(filename_open,"");
  strcat(filename_open,filename_dir);

  // counter
  logLikelihood_nan = 0;

  /* Set initial values for parameters */
  /* results of Zurlinden: */
  // means of posterior distribution
  kk = 1;
  Ith(posteriorZurlindenMean,kk++) = 0.332; Ith(posteriorZurlindenMean,kk++) = 0.0476;
  Ith(posteriorZurlindenMean,kk++) = 123.0; Ith(posteriorZurlindenMean,kk++) = 2.57;
  Ith(posteriorZurlindenMean,kk++) = 1.2e+3; Ith(posteriorZurlindenMean,kk++) = 478.0; Ith(posteriorZurlindenMean,kk++) = 0.345; Ith(posteriorZurlindenMean,kk++) = 467.0;
  Ith(posteriorZurlindenMean,kk++) = 6.14e+3; Ith(posteriorZurlindenMean,kk++) = 4.99e+4; Ith(posteriorZurlindenMean,kk++) = 0.343; Ith(posteriorZurlindenMean,kk++) = 5.21e+3;
  Ith(posteriorZurlindenMean,kk++) = 1.75e+4; Ith(posteriorZurlindenMean,kk++) = 3.54e+4; Ith(posteriorZurlindenMean,kk++) = 2.23e+4; Ith(posteriorZurlindenMean,kk++) = 1.4e+7;
  Ith(posteriorZurlindenMean,kk++) = 3.6e+4; Ith(posteriorZurlindenMean,kk++) = 3.66e+3;
  Ith(posteriorZurlindenMean,kk++) = 0.0123; Ith(posteriorZurlindenMean,kk++) = 0.155; Ith(posteriorZurlindenMean,kk++) = 0.138;
  // coefficitent of variation of posterior distribution
  kk = 1;
  Ith(posteriorZurlindenCoV,kk++) = 0.36; Ith(posteriorZurlindenCoV,kk++) = 0.3;
  Ith(posteriorZurlindenCoV,kk++) = 0.29; Ith(posteriorZurlindenCoV,kk++) = 0.87;
  Ith(posteriorZurlindenCoV,kk++) = 0.39; Ith(posteriorZurlindenCoV,kk++) = 0.25; Ith(posteriorZurlindenCoV,kk++) = 0.4; Ith(posteriorZurlindenCoV,kk++) = 0.38;
  Ith(posteriorZurlindenCoV,kk++) = 0.33; Ith(posteriorZurlindenCoV,kk++) = 0.26; Ith(posteriorZurlindenCoV,kk++) = 0.38; Ith(posteriorZurlindenCoV,kk++) = 0.31;
  Ith(posteriorZurlindenCoV,kk++) = 0.27; Ith(posteriorZurlindenCoV,kk++) = 0.45; Ith(posteriorZurlindenCoV,kk++) = 0.2; Ith(posteriorZurlindenCoV,kk++) = 1.65;
  Ith(posteriorZurlindenCoV,kk++) = 0.87; Ith(posteriorZurlindenCoV,kk++) = 0.69;
  Ith(posteriorZurlindenCoV,kk++) = 0.35; Ith(posteriorZurlindenCoV,kk++) = 0.22; Ith(posteriorZurlindenCoV,kk++) = 0.28;

  sigmaRW_scale = 0.5; // persönlicher Skalierungsfaktor für sigma von proposal

  /* nehme zum Testen die Mittelwerte aus Tabelle 6, S. 275.
  * d.h. ich bin schon im log-transformierten Bereich.
  */
  for (i = 0; i < NSTUDIES; i++) {
    kk=1;
    // Startwerte ein wenig auslenken: ein sigma weit weg (man könnte noch ein random * (+-1) einbauen, damit die Startwerte random links oder rechts vom Zielwert sind.)
    Ith(thetaCurr,i*NPARSTUDY+1) = log(0.332) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // hour (h)
    Ith(thetaCurr,i*NPARSTUDY+2) = log(0.0476) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // hour (h)
    Ith(thetaCurr,i*NPARSTUDY+3) = log(123.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+4) = log(2.57) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;     // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
    Ith(thetaCurr,i*NPARSTUDY+5) = log(1.2e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+6) = log(478.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+7) = log(0.345) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // unitless
    Ith(thetaCurr,i*NPARSTUDY+8) = log(467.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
    Ith(thetaCurr,i*NPARSTUDY+9) = log(6.14e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+10) = log(4.99e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+11) = log(0.343) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // unitless
    Ith(thetaCurr,i*NPARSTUDY+12) = log(5.21e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
    Ith(thetaCurr,i*NPARSTUDY+13) = log(1.75e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+14) = log(3.54e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // micromol per hour (\mymol/h)
    Ith(thetaCurr,i*NPARSTUDY+15) = log(2.23e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
    Ith(thetaCurr,i*NPARSTUDY+16) = log(1.4e+7) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // micromol per hour (\mymol/h)
    Ith(thetaCurr,i*NPARSTUDY+17) = log(3.6e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // per hour (1/h)
    Ith(thetaCurr,i*NPARSTUDY+18) = log(3.66e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // per hour (1/h)
    Ith(thetaCurr,i*NPARSTUDY+19) = log(0.0123) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // liter per hour per BW^{0.75} (L/h/BW^{0.75})
    Ith(thetaCurr,i*NPARSTUDY+20) = log(0.155) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
    Ith(thetaCurr,i*NPARSTUDY+21) = log(0.138) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
    // printf("para1=%g\n", Ith(thetaCurr,i*NPARSTUDY+1));
      // Ith(thetaCurr,i*NPARSTUDY+1) = log(0.332);    // hour (h)
      // Ith(thetaCurr,i*NPARSTUDY+2) = log(0.0476);   // hour (h)
      // Ith(thetaCurr,i*NPARSTUDY+3) = log(123.0);    // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+4) = log(2.57);     // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
      // Ith(thetaCurr,i*NPARSTUDY+5) = log(1.2e+3);   // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+6) = log(478.0);    // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+7) = log(0.345);    // unitless
      // Ith(thetaCurr,i*NPARSTUDY+8) = log(467.0);    // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
      // Ith(thetaCurr,i*NPARSTUDY+9) = log(6.14e+3);  // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+10) = log(4.99e+4);  // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+11) = log(0.343);    // unitless
      // Ith(thetaCurr,i*NPARSTUDY+12) = log(5.21e+3);  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
      // Ith(thetaCurr,i*NPARSTUDY+13) = log(1.75e+4);  // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+14) = log(3.54e+4);  // micromol per hour (\mymol/h)
      // Ith(thetaCurr,i*NPARSTUDY+15) = log(2.23e+4);  // microM (\myM)
      // Ith(thetaCurr,i*NPARSTUDY+16) = log(1.4e+7);   // micromol per hour (\mymol/h)
      // Ith(thetaCurr,i*NPARSTUDY+17) = log(3.6e+4);   // per hour (1/h)
      // Ith(thetaCurr,i*NPARSTUDY+18) = log(3.66e+3);  // per hour (1/h)
      // Ith(thetaCurr,i*NPARSTUDY+19) = log(0.0123);   // liter per hour per BW^{0.75} (L/h/BW^{0.75})
      // Ith(thetaCurr,i*NPARSTUDY+20) = log(0.155);    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
      // Ith(thetaCurr,i*NPARSTUDY+21) = log(0.138);    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
  }
  Ith(thetaCurr,NPARALL) = log(0.4); // sigma der Likelihood-Funktion
  Ith(thetaCan,NPARALL) = Ith(thetaCurr,NPARALL); // für die Studienblöcke Ith(thetaCan,NPARALL) initialisieren

  /* Create user's data */
  data = (UserData) malloc(sizeof *data); if(check_retval((void *)data, "malloc", 2)) return(1);

  /* Create serial vector of length NEQ for I.C. and abstol */
  y_Sundials = N_VNew_Serial(NEQ); if (check_retval((void *)y_Sundials, "N_VNew_Serial", 0)) return(1);
  abstol = N_VNew_Serial(NEQ); if (check_retval((void *)abstol, "N_VNew_Serial", 0)) return(1);

  /****************************************************************************/
  /************************** Vorbereitungen **********************************/
  /****************************************************************************/
  fprintf(fileLog,"\nacetaminophen problem (Zurlinden 2015) - rank: %d\n\n", world_rank);

  /************************* init SUNDIALS ************************************/
  /* Initialize y_Sundials. Muss unten bei reinit auch so stehen. */
  for (i = 1; i <= NEQ; i++) {
    Ith(y_Sundials,i) = 0.0;
  }
  Ith(y_Sundials,31) = 1.0; // phi_PAPS_liver
  Ith(y_Sundials,32) = 1.0; // phi_UDPGA_liver

  /* Set the scalar relative tolerance */
  reltol = RTOL;
  /* Set the vector absolute tolerance */
  for (i = 1; i <= NEQ; i++) { Ith(abstol,i) = ATOL; }

  /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula */
  cvode_mem = CVodeCreate(CV_BDF);
  if (check_retval((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* First of all specify a pointer to the file where all cvode messages
   * should be directed as the default cvode error handler function is used.
   */
  retval = CVodeSetErrFile(cvode_mem, fileLog);
  if (check_retval(&retval, "CVodeSetErrFile", 1)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y_Sundials'=f(t,y_Sundials), the inital time T0, and
   * the initial dependent variable vector y_Sundials. */
  retval = CVodeInit(cvode_mem, f, T0, y_Sundials);
  if (check_retval(&retval, "CVodeInit", 1)) return(1);

  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */
  retval = CVodeSVtolerances(cvode_mem, reltol, abstol);
  if (check_retval(&retval, "CVodeSVtolerances", 1)) return(1);

  /* Call CVodeRootInit to specify the root function g with 2 components */
//  retval = CVodeRootInit(cvode_mem, 2, g);
//  if (check_retval(&retval, "CVodeRootInit", 1)) return(1);

  /* Set the pointer to user-defined data */
  retval = CVodeSetUserData(cvode_mem, data);
  if(check_retval(&retval, "CVodeSetUserData", 1)) return(1);

  /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(NEQ, NEQ);
  if(check_retval((void *)A, "SUNDenseMatrix", 0)) return(1);

  /* Create dense SUNLinearSolver object for use by CVode */
  LS = SUNLinSol_Dense(y_Sundials, A);
  if(check_retval((void *)LS, "SUNLinSol_Dense", 0)) return(1);

  /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
  retval = CVodeSetLinearSolver(cvode_mem, LS, A);
  if(check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

  /* Set the user-supplied Jacobian routine Jac */
  //retval = CVodeSetJacFn(cvode_mem, Jac);
  //if(check_retval(&retval, "CVodeSetJacFn", 1)) return(1);

  /* Set maximum number of steps to be taken by the solver in its attempt to
     reach the next output time. (Default = 500)*/
  retval = CVodeSetMaxNumSteps(cvode_mem, MXSTEPS);
  if(check_retval(&retval, "CVodeSetMaxNumSteps", 1)) return(1);

  /* In while-loop, call CVode, print results, and test for error.
     Break out of loop when last output time has been reached.  */

  /****************************************************************************/
  /****************************************************************************/
  /***********************        main loop        ****************************/
  /****************************************************************************/
  /****************************************************************************/
  for (kk = 1; kk <= NSTUDIES+1; kk++) {Ith(nAccepted,kk) = 0;}
  onepercent = iterAll/100; waitbar_counter = 0;
  starttime_iterAll = MPI_Wtime();  // stop time of main loop
  printf("waitbar: ");

  for (iterJ = 0; iterJ < iterAll; iterJ++) {
    // waitbar
    if (iterJ == waitbar_counter*onepercent) {
      printf(" %d%%_%d ", waitbar_counter*1, world_rank);
      waitbar_counter++;
    }
    fprintf(fileTimes, "\niter\t%d", iterJ);

    /**************************************************************************/
    /*************************** Block: alle Studien **************************/
    /**************************************************************************/
    logLikelihood_can = 0.0;
    for (studyblock = 0; studyblock < NSTUDIES; studyblock++) {
      fprintf(fileCandidates, "%d\t%d", iterJ, studyblock);

      /************************************************************************/
      /********************* draw new block candidate *************************/
      /************************************************************************/
      // random walk, i.e.: can = curr + randn * stdv;
      // ...corresponding to results of Zurlinden, i.e. use Zurlinden's posterior results as proposal distribution
      /* Krauss,2015,S.7: Each parameter of one block is sampled independently,
      *   such that the proposal covariance matrix becomes diagonal. --> d.h. ich
      *   kann jeden Parameter einzeln random-walken.
      */
      starttime_proposal = MPI_Wtime();

      for (kk = 1; kk <= NPARSTUDY; kk++) {
        Ith(thetaCan,studyblock*NPARSTUDY+kk) = Ith(thetaCurr,studyblock*NPARSTUDY+kk);
        if (iterJ>0) { // bei iterJ==0 den Startwertvektor als "Kandidatenvektor" nehmen, damit ich nicht alles doppelt schreiben muss.
          sigmaRW = sigmaRW_scale * log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk);
          sigmaRW = 1e-1;
          zufallszahl = gsl_ran_gaussian(gslrng,sigmaRW);
          Ith(thetaCan,studyblock*NPARSTUDY+kk) += zufallszahl;
          // printf("st_%d %d %f\n",studyblock,kk,zufallszahl);
        }
        fprintf(fileCandidates, "\t%14.6e", Ith(thetaCan,studyblock*NPARSTUDY+kk));
      }

      endtime_proposal = MPI_Wtime(); cputime_proposal = endtime_proposal-starttime_proposal; // fprintf(fileTimes, "\nproposal\t%d\t%10.6e", studyblock, cputime_proposal);
      /**************************************************************************/
      /********************* posterior of candidate *****************************/
      /**************************************************************************/
      // benötige: Kandidaten, Daten
      logPrior_can = 0.0;

      starttime_ODE = MPI_Wtime();
      // Messdaten:
      retval = getDataInfos(studyblock, &nData, &nAPAP_G_S, data); //liefert passend zum block die passenden Infos: nData = # Messdatenpunkten; nAPAP_G_S = wie viele von APAP und/oder APAP-G und/oder APAP-S sind in den Daten
      // create vector for fitting data
      data2fit = N_VNew_Serial(nData*(nAPAP_G_S+1)); if (check_retval((void *)data2fit, "N_VNew_Serial", 0)) return(1); // +1 weil zusätzlich der Zeitpunkt der Daten gespeichert wird
      // data2fit mit passenden Werten füllen
      // retval = getData(studyblock, data2fit); // liefert passend zum studyblock die Daten, an die gefittet werden soll.
      retval = getDataSynthetic(studyblock, data2fit);

      endtime_getInfos = MPI_Wtime(); cputime_getInfos = endtime_getInfos-starttime_ODE; // fprintf(fileTimes, "\nposteriorGetInfos\t%d\t%10.6e", studyblock, cputime_getInfos);
      /************************* löse ODE System ********************************/
      // create vector to save the solution of the ode system
      odeSolution = N_VNew_Serial(nData*colsODESolu); if (check_retval((void *)odeSolution, "N_VNew_Serial", 0)) return(1); // der Lösungsvektor benötigt nicht den Datenzeitpunkt (im Vgl. zu data2fit)
      for (kk = 1; kk <= nData*colsODESolu; kk++) Ith(odeSolution,kk) = 0.0; // init

      /* reset UserData data according to current iteration, i.e. to current block of thetaCan. */
      data->T_G           = exp(Ith(thetaCan,studyblock*NPARSTUDY+1));   // hour (h)
      data->T_P           = exp(Ith(thetaCan,studyblock*NPARSTUDY+2));   // hour (h)
      data->K_APAP_Mcyp   = exp(Ith(thetaCan,studyblock*NPARSTUDY+3));   // microM (\myM)
      data->V_MCcyp       = exp(Ith(thetaCan,studyblock*NPARSTUDY+4));   // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
      data->K_APAP_Msult  = exp(Ith(thetaCan,studyblock*NPARSTUDY+5));   // microM (\myM)
      data->K_APAP_Isult  = exp(Ith(thetaCan,studyblock*NPARSTUDY+6));   // microM (\myM)
      data->K_PAPS_Msult  = exp(Ith(thetaCan,studyblock*NPARSTUDY+7));   // unitless
      data->V_MCsult      = exp(Ith(thetaCan,studyblock*NPARSTUDY+8));   // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
      data->K_APAP_Mugt   = exp(Ith(thetaCan,studyblock*NPARSTUDY+9));   // microM (\myM)
      data->K_APAP_Iugt   = exp(Ith(thetaCan,studyblock*NPARSTUDY+10));  // microM (\myM)
      data->K_UDPGA_Mugt  = exp(Ith(thetaCan,studyblock*NPARSTUDY+11));  // unitless
      data->V_MCugt       = exp(Ith(thetaCan,studyblock*NPARSTUDY+12));  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
      data->K_APAPG_Mmem  = exp(Ith(thetaCan,studyblock*NPARSTUDY+13));  // microM (\myM)
      data->V_APAPG_Mmem  = exp(Ith(thetaCan,studyblock*NPARSTUDY+14));  // micromol per hour (\mymol/h)
      data->K_APAPS_Mmem  = exp(Ith(thetaCan,studyblock*NPARSTUDY+15));  // microM (\myM)
      data->V_APAPS_Mmem  = exp(Ith(thetaCan,studyblock*NPARSTUDY+16));  // micromol per hour (\mymol/h)
      data->k_synUDPGA    = exp(Ith(thetaCan,studyblock*NPARSTUDY+17));  // per hour (1/h)
      data->k_synPAPS     = exp(Ith(thetaCan,studyblock*NPARSTUDY+18));  // per hour (1/h)
      data->k_APAP_R0     = exp(Ith(thetaCan,studyblock*NPARSTUDY+19));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})
      data->k_APAPG_R0    = exp(Ith(thetaCan,studyblock*NPARSTUDY+20));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})
      data->k_APAPS_R0    = exp(Ith(thetaCan,studyblock*NPARSTUDY+21));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})

      endtime_ODE_IV = MPI_Wtime(); cputime_ODE_IV = endtime_ODE_IV-endtime_getInfos; // fprintf(fileTimes, "\nposteriorInitValues\t%d\t%10.6e", studyblock, cputime_ODE_IV);

      /****** ODE-Löser für nächste MCMC-Iteration reseten/neu initieren, *****/
      /****** weil jetzt erst BODYWEIGHT, D_oral, D_oral_rel bekannt sind. ****/
      /* Initialize y_Sundials = vollständige ODE-Solution */
      for (i = 1; i <= NEQ; i++) {
        Ith(y_Sundials,i) = 0.0;
      }
      if (data->D_oral_rel) {
        Ith(y_Sundials,5) = data->D_oral*data->BW;
      } else {
        Ith(y_Sundials,5) = data->D_oral;
      }
      Ith(y_Sundials,31) = 1.0; // phi_PAPS_liver
      Ith(y_Sundials,32) = 1.0; // phi_UDPGA_liver
      /* reset CVODE with actual y_Sundials */
      retval = CVodeReInit(cvode_mem, T0, y_Sundials);
      if (check_retval(&retval, "CVodeReInit", 1)) return(1);

      endtime_ODE_reinit = MPI_Wtime(); cputime_ODE_reinit = endtime_ODE_reinit-endtime_ODE_IV; // fprintf(fileTimes, "\nposteriorReinitSundials\t%d\t%10.6e", studyblock, cputime_ODE_reinit);

      iout = 1;                     // variables for ode solver: iout counts the # of output times
      /* solve ode system */
      while(1) {
        tout = Ith(data2fit,iout*(nAPAP_G_S+1)+1);          // nächste output time setzen: Datenzeitpunkte sind an den Stellen 1, 1*(nAPAP_G_S+1)+1, 2*(nAPAP_G_S+1)+1, ...
        retval = CVode(cvode_mem, tout, y_Sundials, &t, CV_NORMAL);  // Löse bis zum Zeitpunkt tout

        if (check_retval(&retval, "CVode", 1)) {
          printf("iter %d rank %d ", iterJ, world_rank);
          fprintf(fileLog, "SUNDIALS_ERROR: world_rank %d -> CVode() failed with retval = %d: In iter = %d and studyblock = %d at t = %g from tout = %g\n\n", world_rank, retval, iterJ, studyblock, t, tout);
          endtime_ODE_fail = MPI_Wtime(); fprintf(fileTimes, "\nposteriorODEFailed\t%d\t%10.6e", studyblock, endtime_ODE_fail - endtime_ODE_reinit);
          cputime_ODE_fail += endtime_ODE_fail - endtime_ODE_reinit;
          ode_fail++;
          // printf("candidate_%3d_%2d: \n",iterJ,studyblock);
          // for (i = 1; i <= NPARSTUDY; i++) {
          //   printf("%14.6e ", Ith(thetaCan,studyblock*NPARSTUDY+i));
          // }
          // printf("\n");
          nAcceptBool = -2; // d.h. sundials hatte Fehler
          break;
        }
        if (retval == CV_SUCCESS) {
          // save current step of solution into vector (for likelihood)
          for (kk = 1; kk <= colsODESolu; kk++) {
            Ith(odeSolution,iout*colsODESolu + kk) = Ith(y_Sundials,10*kk); // 10*kk sind die Ergebnisse bzgl. Venenkonzentration. (10, weil es jeweils 10 Kompartimente für APAP,-G,-S sind)
          }
          iout++; // # of output times
        }

        if (iout == nData) break;
      }
      /************************************************************************/
      retval2 = CVodeGetNumSteps(cvode_mem, &nIterSundials);
      // retval2 = CVodeGetNumNonlinSolvIters(cvode_mem, &nIterSundials);
      nIterSundials_sum += nIterSundials;

      endtime_ODE = MPI_Wtime(); cputime_ODE_solve = endtime_ODE - endtime_ODE_reinit; fprintf(fileTimes, "\nposteriorODEsolve\t%d\t%10.6e", studyblock, cputime_ODE_solve);
      if (iterJ>0) {
        cputime_ODE_sum += cputime_ODE_solve;
      }

      if (nAcceptBool==-2) {
        nIterSundials_fail += nIterSundials;
      }

      fprintf(filenIterSundials, "%ld\t%10.4e\t", nIterSundials, (double) cputime_ODE_solve/nIterSundials);
      /************************************************************************/
      /* likelihood berechnen (als LogLikelihood) */
      retval = getLogLikelihoodTheta(nData, nAPAP_G_S, &logLikelihood_can_blockj, &LikelihoodNumerator_blockj, odeSolution, data2fit, exp(Ith(thetaCurr,NPARALL)), LogNormalBool);
      if (studyblock==1) {
        // printf("ndata:%d nAPAP_G_S:%d logLikelihood_can_blockj:%f LikelihoodNumerator_blockj:%f odeSolution... data2fit... sigmaLL:%f LogNormalBool:%d\n", nData, nAPAP_G_S, logLikelihood_can_blockj, LikelihoodNumerator_blockj, exp(Ith(thetaCurr,NPARALL)), LogNormalBool);
      }
      if (iterJ==0) {
        logLikelihood_can += logLikelihood_can_blockj; // verwenden später bei sigmaLL-Block
      } else {
        logLikelihood_can = logLikelihood_can_blockj;
      }

      /************************************************************************/
      /* prior der Kandidaten berechnen (als LogPrior) */
      for (kk = 1; kk <= NPARSTUDY; kk++) {
        Ith(thetaCan_temp,kk) = Ith(thetaCan,studyblock*NPARSTUDY+kk); // thetaCan sind im log-Bereich. Für Priorauswertung in normalen Bereich zurück? Dann müssten Priorverteilungen auch entsprechend für den normalen Bereich sein...
      }
      retval = getLogPrior(&logPrior_can, thetaCan_temp);

      // zähle mal die Likelihoods, die nan sind.
      if (isnan(logLikelihood_can_blockj)){
        logLikelihood_nan++;
      }

      /************************************************************************/
      /* posterior berechnen (LogPosterior) */
      Ith(logPosterior_can,studyblock+1) = logLikelihood_can_blockj + logPrior_can;    // logPosterior_can als Vektor, um bei Akzeptanzrate mit dem passenden Block aus vorherigem Schritt vergleichen zu können

      if (iterJ==0) {
        retval = printSolutionToFile(fileSolution, iterJ, studyblock, nData, data2fit, nAPAP_G_S, odeSolution);
      }

      endtime_posterior = MPI_Wtime();
      cputime_posterior_rest = endtime_posterior-endtime_ODE; // fprintf(fileTimes, "\nposteriorRest\t%d\t%10.6e", studyblock, cputime_posterior_rest);
      cputime_posterior = endtime_posterior-endtime_proposal; fprintf(fileTimes, "\nposterior\t%d\t%10.6e", studyblock, cputime_posterior);

      if (iterJ>0) { // bei j==0 wird noch der theta_Anfangswertvektor ausgewertet
        /************************************************************************/
        /****************** akzeptanzrate berechnen *****************************/
        /************************************************************************/
        logaccrate = Ith(logPosterior_can,studyblock+1) - Ith(logPosterior_curr,studyblock+1); // in dieser Form, solange Proposal symmetrisch ist! Sonst nochmal ändern!

        /************************************************************************/
        /****************** accept-reject Schritt *******************************/
        /************************************************************************/
        zufallszahl = log(gsl_rng_uniform(gslrng));
        // printf("st_%d acc %f\n",studyblock,zufallszahl);
        logaccrateUniform = zufallszahl;                        // LogUniform, um mit LogPosterior vergleichen zu können!

        if(logaccrateUniform < logaccrate) { // accept
          Ith(nAccepted,studyblock+1) += 1;
          nAcceptBool = 1;

          fprintf(fileMarkovChain, "%d\t%d", iterJ, studyblock);
          for (kk = 1; kk <= NPARSTUDY; kk++) {
            // save candidate into current
            Ith(thetaCurr,studyblock*NPARSTUDY+kk) = Ith(thetaCan,studyblock*NPARSTUDY+kk);
            // write thetaCurr to file
            fprintf(fileMarkovChain, "\t%14.6e", Ith(thetaCurr,studyblock*NPARSTUDY+kk));
          }
          fprintf(fileMarkovChain, "\n");

          retval = printSolutionToFile(fileSolution, iterJ, studyblock, nData, data2fit, nAPAP_G_S, odeSolution);

          Ith(logPosterior_curr,studyblock+1)   = Ith(logPosterior_can,studyblock+1);
          Ith(LikelihoodNumerator,studyblock+1) = LikelihoodNumerator_blockj;   // für sigmaLL-Block abspeichern, damit dort das ODE-System nicht nochmal gelöst werden muss
        } else { // reject
          if (nAcceptBool!=-2) {                                                // ist -2, falls sundials einen Fehler hatte. Dann wird der Block (hoffentlich) nicht akzeptiert worden sein.
            nAcceptBool = 0;                                                    // für fprintf(fileMarkovChain,...) im sigmaLL-Block
          }
        }
        fprintf(fileAcceptReject, "%14.6e\t%14.6e\t%d\n",logaccrateUniform, logaccrate, nAcceptBool);
        endtime_accrej = MPI_Wtime(); cputime_accrej = endtime_accrej-endtime_posterior; // fprintf(fileTimes, "\naccrej\t%d\t%10.6e", studyblock, cputime_accrej);

      } else { // iterJ == 0
        if (studyblock==0){
          for (i = 0; i < NSTUDIES; i++) {
            fprintf(fileMarkovChain, "%d\t%d", iterJ, i);
            for (kk = 1; kk <= NPARSTUDY; kk++) {fprintf(fileMarkovChain, "\t%14.6e", Ith(thetaCurr,i*NPARSTUDY+kk));}
            fprintf(fileMarkovChain, "\n");
          }
        }
        Ith(logPosterior_curr,studyblock+1)   = Ith(logPosterior_can,studyblock+1);
        Ith(LikelihoodNumerator,studyblock+1) = LikelihoodNumerator_blockj;     // für sigmaLL-Block abspeichern, damit dort das ODE-System nicht nochmal gelöst werden muss
        nAcceptBool = 1;
      }
      fprintf(fileCandidates, "\t%d\n", nAcceptBool);

      nAcceptBool=0;                                                            // immer zurücksetzen, damit er in  nächster Iteration nicht mehr auf -2 steht.
    } // for studyblock=0,...,NSTUDIES

    /**************************************************************************/
    /**************************** Block: sigmaLL ******************************/
    /**************************************************************************/
    starttime_sigmaLL = MPI_Wtime();

    /*********/
    /** draw candidate: nachher, damit die Studienblöcke in der nächsten Iteration  **/
    /**/

    Ith(thetaCan,NPARALL) = Ith(thetaCurr,NPARALL);
    if (iterJ>0){
      // sigmaLL würfeln
      // Ith(thetaCan,NPARALL) = 0.5+gsl_rng_uniform(gslrng);
      zufallszahl = gsl_ran_gaussian(gslrng,1e-3);
      Ith(thetaCan,NPARALL) += zufallszahl; // würfele sigmaLL mittels Random Walk
    }
    fprintf(fileCandidates, "%d\t10", iterJ);
    fprintf(fileCandidates, "\t%14.6e", Ith(thetaCan,NPARALL));

    /**************************************************************************/
    /********************* posterior of candidate *****************************/
    /**************************************************************************/
    // Likelihood: Daten, ODE-Lösung, sigmaLL
    if (iterJ==0){
      // nichts zu tun, weil logLikelihood_can noch von oben passend ist
    } else {
      logLikelihood_can = 0.0;
      for (studyblock = 0; studyblock < NSTUDIES; studyblock++) {
        // Messdaten:
        retval = getDataInfos(studyblock, &nData, &nAPAP_G_S, data); //liefert passend zum block die passenden Infos: nData = # Messdatenpunkten; nAPAP_G_S = wie viele von APAP und/oder APAP-G und/oder APAP-S sind in den Daten
        // create vector for fitting data of current studyblock
        data2fit = N_VNew_Serial(nData*(nAPAP_G_S+1)); if (check_retval((void *)data2fit, "N_VNew_Serial", 0)) return(1); // +1 weil zusätzlich der Zeitpunkt der Daten gespeichert wird
        // data2fit mit passenden Werten füllen
        // retval = getData(studyblock, data2fit); // liefert passend zum studyblock die Daten, an die gefittet werden soll.
        retval = getDataSynthetic(studyblock, data2fit);

        /************************* löse ODE System ****************************/

        /* likelihood berechnen (als LogLikelihood) */
        retval = getLogLikelihoodSigma(nData, nAPAP_G_S, &logLikelihood_can_blockj, LikelihoodNumerator, studyblock+1, data2fit, exp(Ith(thetaCan,NPARALL)), LogNormalBool);
        // if (studyblock==1) {
        //   printf("\tndata:%d nAPAP_G_S:%d logLikelihood_can_blockj:%f LikelihoodNumerator_blockj:... studyblock:%d data2fit... sigmaLL:%f LogNormalBool:%d\n", nData, nAPAP_G_S, logLikelihood_can_blockj, studyblock, exp(Ith(thetaCan,NPARALL)), LogNormalBool);
        // }
        logLikelihood_can += logLikelihood_can_blockj;
      }
    } // else: iterJ>0

    /* Prior von sigmaLL */
    // logPrior_can -= 2.0*log(exp(Ith(thetaCan,NPARALL))); // Jeffrey: 1/sigma^2 .
    logPrior_can = - 0.5*log(2*M_PI) - log(1*exp(Ith(thetaCan,NPARALL))) - pow(Ith(thetaCan,NPARALL) +1,2) / (2*1); // log( logNormal(-1,1) )

    /******************* posterior berechnen (LogPosterior) *******************/
    Ith(logPosterior_can,NSTUDIES+1) = logLikelihood_can + logPrior_can;

    if (iterJ>0) { // bei j==0 wird noch der theta_Anfangswertvektor ausgewertet
      /************************************************************************/
      /****************** akzeptanzrate berechnen *****************************/
      /************************************************************************/
      logaccrate = Ith(logPosterior_can,NSTUDIES+1) - Ith(logPosterior_curr,NSTUDIES+1); // in dieser Form, solange Proposal symmetrisch ist! Sonst nochmal ändern!

      /************************************************************************/
      /****************** accept-reject Schritt *******************************/
      /************************************************************************/
      zufallszahl = log(gsl_rng_uniform(gslrng));
      logaccrateUniform = zufallszahl;                          // LogUniform, um mit LogPosterior vergleichen zu können!
      // printf("sigma acc %f\n",zufallszahl);

      if(logaccrateUniform < logaccrate) { // accept
        Ith(nAccepted,NSTUDIES+1) += 1;
        nAcceptBool = 1;
        // save candidate into current
        Ith(thetaCurr,NPARALL) = Ith(thetaCan,NPARALL);
        // write thetaCurr to file
        fprintf(fileMarkovChain, "%d\t%d\t%14.6e\n", iterJ, 10, Ith(thetaCurr,NPARALL));

        // save logPosterior_can into _curr
        Ith(logPosterior_curr,NSTUDIES+1) = Ith(logPosterior_can,NSTUDIES+1);
      } else { // reject
        nAcceptBool = 0;
      }
      fprintf(fileAcceptReject, "%14.6e\t%14.6e\t%d\n",logaccrateUniform, logaccrate, nAcceptBool);

    } else { // iterJ == 0
      fprintf(fileMarkovChain, "%d\t%d\t%14.6e\n", iterJ, 10, Ith(thetaCurr,NPARALL));
      Ith(logPosterior_curr,NSTUDIES+1) = Ith(logPosterior_can,NSTUDIES+1);
      nAcceptBool = 1;
    }
    /*************************** Block sigmaLL ENDE ***************************/
    endtime_sigmaLL = MPI_Wtime(); fprintf(fileTimes, "\nsigmaLL-Block\t%10.6e", endtime_sigmaLL-starttime_sigmaLL);

    fprintf(fileCandidates, "\t%d\n", nAcceptBool);

    if (iterJ>0) {
      /************************************************************************/
      /****************************** swap chains *****************************/
      /************************************************************************/
      // reset cputimes to 0.0
      cputime_swap_toss=0.0; cputime_swap_bcast=0.0; cputime_swap_barrier=0.0;
      cputime_swap_partner=0.0; cputime_swap_wait=0.0;

      if ((world_size>1)&&(swaps>1)) {
        if (world_rank==RANKMASTER) {
          int swapscheme[swaps_max];

          retval = getSwapscheme(swaps_max, swaps, gslrng, swapscheme);
          endtime_swap_toss = MPI_Wtime(); cputime_swap_toss = endtime_swap_toss-endtime_accrej; fprintf(fileTimes, "\nswaptoss\t%10.6e", cputime_swap_toss);

          // sende jeder (außer RANKMASTER selbst) Kette ihren Tauschpartner (broadcast)
          for (i = 0; i < world_size; i++) {
            if (i!=RANKMASTER) {
              MPI_Send(&swapscheme[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD);
            } else {
              swapPartner = swapscheme[i];
            }
          }
          endtime_swap_bcast = MPI_Wtime(); cputime_swap_bcast = endtime_swap_bcast-endtime_swap_toss; fprintf(fileTimes, "\nswapbcast\t%10.6e", cputime_swap_bcast);

        } else { // if world_rank==RANKMASTER
          // Empfange vom RANKMASTER den Tauschpartner
          MPI_Recv(&swapPartner, 1, MPI_INT, RANKMASTER, tag, MPI_COMM_WORLD, &status);
          endtime_swap_bcast = MPI_Wtime(); cputime_swap_bcast = endtime_swap_bcast-endtime_accrej; fprintf(fileTimes, "\nswapbcast\t%10.6e", cputime_swap_bcast);
        }

        // endtime_swap_bcast = MPI_Wtime(); cputime_swap_bcast = endtime_swap_bcast-endtime_accrej; fprintf(fileTimes, "\nswapbcast\t%10.6e", cputime_swap_bcast);
        MPI_Barrier(MPI_COMM_WORLD);
        endtime_swap_barrier = MPI_Wtime(); cputime_swap_barrier = endtime_swap_barrier-endtime_swap_bcast; fprintf(fileTimes, "\nswapbarrier\t%10.6e", cputime_swap_barrier);

        /* perform the swap */
        if (world_rank!=swapPartner) {
          // sende thetaCurr, logPosterior_curr an swapPartner
          for (i = 1; i <= NPARALL; i++) {
            thetaCurr_swap_send[i-1] = Ith(thetaCurr,i);
            thetaCurr_swap_recv[i-1] = 0.0;
          }
          for (i = 1; i <= NSTUDIES+1; i++) {
            logPosterior_swap_send[i-1] = Ith(logPosterior_curr,i);
          }

          // printf("%d: world_rank %d before MPI_Isend to %d\n", iterJ, world_rank, swapPartner);
          MPI_Isend(thetaCurr_swap_send, NPARALL, MPI_DOUBLE, swapPartner, tag+1, MPI_COMM_WORLD, &request_swap_theta);
          MPI_Isend(logPosterior_swap_send, NSTUDIES+1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &request_swap_logPosterior);
          // printf("%d: world_rank %d after MPI_Isend to %d\n", iterJ, world_rank, swapPartner);
          // empfange selbiges vom swapPartner
          MPI_Recv(thetaCurr_swap_recv, NPARALL, MPI_DOUBLE, swapPartner, tag+1, MPI_COMM_WORLD, &status_swap_theta);
          MPI_Recv(logPosterior_swap_recv, NSTUDIES+1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &status_swap_logPosterior);
          endtime_swap_partner = MPI_Wtime(); cputime_swap_partner = endtime_swap_partner-endtime_swap_barrier; fprintf(fileTimes, "\nswappartner\t%10.6e", cputime_swap_partner);

          MPI_Wait(&request_swap_theta, &status_swap_theta);
          MPI_Wait(&request_swap_logPosterior, &status_swap_logPosterior);
          endtime_swap_wait = MPI_Wtime(); cputime_swap_wait = endtime_swap_wait-endtime_swap_partner; fprintf(fileTimes, "\nswapwait\t%10.6e", cputime_swap_wait);

          // speichere empfangene Daten in entsprechenden Variablen
          for (i = 1; i <= NPARALL; i++) {
            Ith(thetaCurr,i) = thetaCurr_swap_recv[i-1];
          }
          for (i = 1; i <= NSTUDIES+1; i++) {
            Ith(logPosterior_curr,i) = logPosterior_swap_recv[i-1];
          }
        }
      } // if world_size > 1
      endtime_swap = MPI_Wtime(); fprintf(fileTimes, "\nswap\t%10.6e", endtime_swap - endtime_accrej);
      cputime_swap += endtime_swap - endtime_accrej;

      /****************************** end of swap *****************************/
    }

    fprintf(filenIterSundials, "\n");
  } // end of 'iterAll'
  printf("100%%\n\n");

  endtime_iterAll = MPI_Wtime();
  cputime_iterAll = endtime_iterAll - starttime_iterAll;

  fprintf(fileTimes, "\ncputimeIterAll\t%10.6e", cputime_iterAll);
  fprintf(filenIterSundials, "%d\t%10.6e\t%ld\t%10.6e\t%ld\t%10.6e\n", seedBool, cputime_iterAll, nIterSundials_sum, (double) cputime_ODE_sum/nIterSundials_sum, nIterSundials_fail, (double) cputime_ODE_fail/nIterSundials_fail);

  /****************************************************************************/
  /************************ end of main loop **********************************/
  /****************************************************************************/

  // fprintf(fileLog,"world_rank_%d:\titerAll:\t%d\tcputime_iterAll:\t%g\n", world_rank, iterAll, cputime_iterAll);
  // fprintf(fileLog,"world_rank_%d:\tcputime_ODE_sum:\t%g\t_mean:\t%g\n", world_rank, cputime_ODE_sum, cputime_ODE_sum/iterAll/NSTUDIES);
  // fprintf(fileLog,"world_rank_%d:\tcputime_ODE_fail:\t%g\tmean:\t%g\t#ode fail:\t%d\n", world_rank, cputime_ODE_fail, (double) cputime_ODE_fail/(ode_fail), ode_fail);
  // fprintf(fileLog,"world_rank_%d:\tcputime_iterAll-ODE_fail:\t%g\n", world_rank, cputime_iterAll-cputime_ODE_fail);
  // fprintf(fileLog,"world_rank_%d:\tcputime_swap_sum:\t%g\t_mean:\t%g\n", world_rank, cputime_swap, cputime_swap/iterAll);
  // fprintf(fileLog,"world_rank_%d:\tlogLikelihood_nan:\t%d\n", world_rank, logLikelihood_nan);

  fprintf(fileLog, "world_rank\t%d\naccepted:", world_rank);
  acceptrate = 0.0;
  for (kk = 1; kk <= NSTUDIES+1; kk++) {
    fprintf(fileLog,"\t%d", kk);
  }
  fprintf(fileLog, "\n\t");
  for (kk = 1; kk <= NSTUDIES+1; kk++) {
    acceptrate += Ith(nAccepted,kk);
    fprintf(fileLog,"\t%d", (int)Ith(nAccepted,kk));
  }
  fprintf(fileLog, "\n\t");
  for (kk = 1; kk <= NSTUDIES+1; kk++) {
    fprintf(fileLog,"\t%g", Ith(nAccepted,kk)/iterAll);
  }
  fprintf(fileLog, "\n");
  acceptrate = (double) acceptrate/iterAll/NSTUDIES;
  fprintf(fileLog, "accrate\t\t%g\n", acceptrate);
  fprintf(fileLog, "Zeiten\t\tZeiten\t\tZeiten\t\tZeiten\t\tZeiten\t\tnIter\n");
  fprintf(fileLog, "\t\tODE\t\t\t\t\t\tKommunikation\tSundials\n");
  fprintf(fileLog, "overall\t\tsum\t\tfail\t\tsum-fail\tswap\t\tall\tfail\tsum-fail\n");
  fprintf(fileLog, "%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%ld\t%ld\t%ld\n", cputime_iterAll, cputime_ODE_sum, cputime_ODE_fail, cputime_ODE_sum-cputime_ODE_fail, cputime_swap, nIterSundials_sum, nIterSundials_fail, nIterSundials_sum-nIterSundials_fail);
  // fprintf(fileLog, "%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%ld\t%ld\t%ld\n", cputime_iterAll, cputime_ODE_sum, cputime_ODE_fail, cputime_ODE_sum-cputime_ODE_fail, cputime_bcast_sigmaLL_sum, cputime_gather_LLNumer_sum, cputime_swap, nIterSundials_sum, nIterSundials_fail, nIterSundials_sum-nIterSundials_fail);

  /* Free y_Sundials and abstol vectors */
  N_VDestroy(y_Sundials);
  N_VDestroy(abstol);

  /* Free integrator memory */
  CVodeFree(&cvode_mem);

  /* Free the linear solver memory */
  SUNLinSolFree(LS);

  /* Free the matrix memory */
  SUNMatDestroy(A);

  // close files
  fclose(fileSolution);
  fclose(fileMarkovChain);
  fclose(fileAcceptReject);
  fclose(fileCandidates);

  // close MPI
  MPI_Finalize();

  return(retval);
}
