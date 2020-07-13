/* -----------------------------------------------------------------
 * blockmultMH:
 *
 * Der blockweise Metropolis-Hastings mit symmetrischem Random-Walk
 * für jeden Block als Proposalverteilung und je nach Übergabeparameter
 * mehrkettig und/oder mit Positionstausch.
 * Je nach Übergabeparameter werden mehrere Blöcke von einem Rank
 * bearbeitet.
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
// #include <sys/time.h>

#include <gsl/gsl_rng.h>
#include <gsl-sprng.h>
#include <gsl/gsl_randist.h>
// #include <gsl/gsl_math.h>
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
#define NEQ   32               /* number of equations of ODE System  */
//#define AMT   RCONST(5.0e+3)   /* amount of drug dosis */
#define RTOL  RCONST(1.0e-6)   /* scalar relative tolerance            */
#define ATOL  RCONST(1.0e-14)  /* vector absolute tolerance components e-8,e-10 */
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
#define RANKMASTER 10

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
  double starttime_bcast_sigmaLL, endtime_bcast_sigmaLL, endtime_gather_LLNumer;
  double endtime_swap_toss, endtime_swap_bcast, endtime_swap_bcast_master, endtime_swap_barrier, starttime_swap_barrier, endtime_swap_partner, endtime_swap_wait, endtime_swap;
  double cputime_ODE, cputime_ODE_sum=0.0, cputime_iterAll, cputime_swap=0.0;
  double cputime_proposal=0.0, cputime_posterior=0.0, cputime_accrej=0.0;
  double cputime_getInfos=0.0, cputime_ODE_solve=0.0, cputime_ODE_IV=0.0, cputime_ODE_reinit=0.0, cputime_ODE_fail=0.0, cputime_posterior_rest=0.0;
  double cputime_swap_toss=0.0, cputime_swap_bcast=0.0, cputime_swap_bcast_master=0.0, cputime_swap_barrier=0.0, cputime_swap_partner=0.0, cputime_swap_wait=0.0;
  double cputime_bcast_sigmaLL=0.0, cputime_bcast_sigmaLL_sum=0.0, cputime_gather_LLNumer=0.0, cputime_gather_LLNumer_sum=0.0;
  double starttime_swap;
  double acceptrate, sigmaLL_Can_next, sigmaLL_AccRej_next, param_Can, param_Acc;
  double *thetaCurr_swap_send, *thetaCurr_swap_recv, *logPosterior_curr_block_send, *logPosterior_curr_block_recv;
  realtype reltol, t, tout;
  N_Vector y_Sundials, abstol, thetaCurr, thetaCan, thetaCan_temp, odeSolution, posteriorZurlindenMean, posteriorZurlindenCoV;
  N_Vector data2fit, LikelihoodNumerator, logPosterior_can_block, logPosterior_curr_block, nAccepted;
  UserData data;
  SUNMatrix A;
  SUNLinearSolver LS;
  void *cvode_mem;
  int retval=0, retval2=0, iout, i, ctr, iterJ, k, kk, nData, nAPAP_G_S, studyblock, colsODESolu=3, LogNormalBool=0;
  int iterAll, swaps, seedBool, multipleChainsBool, blocksPerRank;
  int swaps_max;
  int *swapscheme;
  int masterrank, world_rank, world_size, tag=42, color_split, chainmaster, chain_rank, chain_size, nChains, swapPartner, world_chainnumber, world_chainmaster, nClients;
  int rootsfound[2], logLikelihood_nan, ode_fail=0, waitbar, waitbar_counter, onepercent, nAcceptBool;
  long nIterSundials=0, nIterSundials_sum=0, nIterSundials_fail=0;
  double x,can,logaccrate,logaccrateUniform, logPosterior_can_sigma, logPosterior_curr_sigma;
  double *sigmaRW_send, *sigmaRW_recv;
  int *sendcounts, *displs, recvcount;
  realtype mu, sigmaRW, sigmaRW_scale, max, min, thetaCanI;
  realtype logPrior_can, logPrior_can_temp, logLikelihood_can, logLikelihood_can_blockj, logPosterior, LikelihoodNumerator_blockj_curr, LikelihoodNumerator_blockj_can, sigmaLL;
  realtype logPosterior_swap_sigma, sigmaLL_send, sigmaLL_recv;
  gsl_rng *gslrng;
  FILE *fileLog, *fileSolution, *fileMarkovChain, *fileAcceptReject, *fileCandidates, *fileTimes, *filenIterSundials;
  char filename[40], filename_dir[50], filename_open[100];
  unsigned long seed;
  MPI_Status status, status_swap_theta, status_swap_logPosterior;
  MPI_Request request_bcast, request_swap_logPosterior, request_swap_theta, request_sigmaRW;
  MPI_Comm *new_comm, chain_comm, unneeded_comm;

  // init MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

  /* get inputs */
  // Constant for MCMC loop
  if (argc>1) {
    iterAll = atoi(argv[1]);
    switch (iterAll) {
      case 2: break;
      case 10: break;
      case 100: break;
      case 1000: break;
      case 10000: break;
      case 100000: break;
      case 1000000: break;
      default: printf("Improper input for iterAll.\n Take one out of 2, 10, 100, 1000, 10000, 100000, 1000000.\n"); MPI_Finalize(); return(1); break;
    }
  } else {
    printf("A constant (int) for # of iterations is needed.\nEnd.\n");
    MPI_Finalize();
    return(1);
  }
  // Constant for swap
  if (argc>2) {
    swaps = atoi(argv[2]);
    if (swaps<0) {
      printf("Improper input for swaps.\nTake a non-negative number.\n");
      MPI_Finalize();
      return(1);
    }
  } else {
    swaps = 0;
    printf("No input for # of swapping chains is given -> we work without swaps.\n");
  }
  // Constants for seed
  if (argc>3) {
    seedBool = atoi(argv[3]);
  } else {
    seedBool = 0;
    printf("No input for boolean seed. Take default seed.\n");
  }
  // bool if multiple chains are asked
  if (argc>4) {
    multipleChainsBool = atoi(argv[4]);
    switch (multipleChainsBool) {
      case 0: break;
      case 1: break;
      default: printf("Improper input for multipleChainsBool (mChainsBool).\nTake one out of 0,1.\n"); MPI_Finalize(); return(1); break;
    }
  } else {
    multipleChainsBool = 0;
    printf("No specification is done for multiple chains -> no multiple, only one chain.\n");
  }
  // defines how many studyblocks should be treated by one rank
  if (argc>5) {
    blocksPerRank = atoi(argv[5]);
    switch (blocksPerRank) {
      case 1: break;
      case 2: break;
      case 5: break;
      case 10: break;
      default: printf("Improper input for blocksPerRank (bPR).\nTake one out of 1,2,5,10.\n"); MPI_Finalize(); return(1); break;
    }
  } else {
    blocksPerRank = 10;
    printf("\nNo specification is done about the number of blocks per rank = 1,2,5,10.\n We will work with no block-parallelisation: bPR = %d.\n", blocksPerRank);
  }


  /* setup for multiple independent chains
   *
   * possible ways:
   * a) split communicator: chap.6.4 (p.235)->chap.6.4.2 (p.244) in mpi-report: https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf
   *                        or: https://www.codingame.com/playgrounds/349/introduction-to-mpi/splitting
   *                        or: https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
   *    -> Beachte: MPI has a limited number of objects that it can create at a time and not freeing your objects could result in a runtime error if MPI runs out of allocatable objects.
   * b) spawn .c file multiple times: chap.10.3 (p.374) in mpi-report: https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf
   */
  // way a):
  chain_size = NSTUDIES/blocksPerRank + 1;

  masterrank = chain_size-1; // ist 10=RANMASTER, wenn alles wie bisher; und sonst genau der zustätzliche rank für den Master
  nClients = chain_size-1;

  // check # of processors
  if (world_size < chain_size) {
    if (world_rank==0) {
      printf("\nToo few processors initialized: %d.\nWe need at least %d. (Better: a multiple of %d)\nEnd.\n", world_size, chain_size,chain_size);
    }
    MPI_Finalize();
    return(1);
  }
  if (world_rank==0){
    printf("You asked for %d processors. Let's see.\n", world_size);
  }

  /* check if multiple chains are asked */
  // Calculate variables
  if (multipleChainsBool) {
    nChains = world_size/(chain_size);
    world_chainnumber = world_rank/(chain_size);
  } else {
    // nur eine Kette - alle überflüssigen ranks werden unten aussortiert (d.h. alle mit rank>=11)
    nChains = 1;
    world_chainnumber = 0;
  }


  /*
   * split MPI_COMM_WORLD into nChains-many communicators.
   */
  if (world_rank>=(nChains*(chain_size))) {
    if (multipleChainsBool==1) {
      printf("world_rank_%d will be ignored: We work with a multiple of %d processors.\n", world_rank, chain_size);
    } else {
      printf("world_rank_%d will be ignored: Only one chain is asked.\n", world_rank);
    }
    color_split = MPI_UNDEFINED;
    new_comm = &unneeded_comm;
  } else {
    color_split = world_rank/(chain_size);
    new_comm = &chain_comm;
  }
  MPI_Comm_split(MPI_COMM_WORLD, color_split, world_rank, new_comm);

  if (color_split!=MPI_UNDEFINED) { // lasse die überflüssigen Prozessoren außer acht.
    MPI_Comm_rank(chain_comm, &chain_rank);
    // MPI_Comm_size(chain_comm, &chain_size);
    world_chainmaster = world_chainnumber*chain_size+masterrank;                // ist der Kettenmaster bzgl. world_rank
    if (chain_rank==masterrank) {
      printf("Here is world_rank %d with a chain of size %d.\nLet's start.\n\n", world_rank, chain_size);
    }

    // Constants for swap
    swaps_max = nChains;
    if (swaps > swaps_max) {
      swaps = swaps_max;
      printf("We were asked for more chains to swap than possible --> set it onto max. possible: %d\n", swaps);
    }
    if (swaps%2==1) {
      printf("# for swaps is uneven: %d", swaps);
      swaps--;
      printf(" --> we work with %d\n", swaps);
    }

    // create log-files
    sprintf(filename_dir,""); sprintf(filename_open,""); sprintf(filename_dir,"");

    sprintf(filename_dir,"output/blockmultMH/%d/%d_%d_",iterAll,world_size,swaps);
    strcat(filename_open,filename_dir);

    sprintf(filename,"log_%d.txt",world_rank);
    strcat(filename_open,filename);
    fileLog=fopen(filename_open,"a"); // append to file
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

    // write into log-file
    fprintf(fileLog, "We have %d ranks initialized.\n", world_size);
    fprintf(fileLog, "We have %d iterations initialized.\n", iterAll);
    fprintf(fileLog, "We perform the swap between %d chains.\n", swaps);
    fprintf(fileLog, "We started %d chains with a size of %d.\n", nChains, chain_size);
    fprintf(fileLog, "We treat %d blocks by one rank.\n", blocksPerRank);

    // preallocate and create vector for parameters
    thetaCurr     = NULL;
    thetaCan      = NULL;
    // thetaCan_temp = NULL;
    nAccepted = NULL;
    if (chain_rank==masterrank) {
      thetaCurr     = N_VNew_Serial(1);
      thetaCan      = N_VNew_Serial(1);
      // thetaCan_temp = N_VNew_Serial(1);
      nAccepted     = N_VNew_Serial(1);
      LikelihoodNumerator = NULL; LikelihoodNumerator = N_VNew_Serial(NSTUDIES); if (check_retval((void *)LikelihoodNumerator, "N_VNew_Serial", 0)) return(1);
    } else {
      thetaCurr     = N_VNew_Serial(blocksPerRank*NPARSTUDY);
      thetaCan      = N_VNew_Serial(NPARSTUDY);
      // thetaCan_temp = N_VNew_Serial(NPARSTUDY);
      nAccepted     = N_VNew_Serial(blocksPerRank);
      posteriorZurlindenMean  = NULL; posteriorZurlindenMean = N_VNew_Serial(NPARSTUDY); if (check_retval((void *)posteriorZurlindenMean, "N_VNew_Serial", 0)) return(1);
      posteriorZurlindenCoV   = NULL; posteriorZurlindenCoV = N_VNew_Serial(NPARSTUDY); if (check_retval((void *)posteriorZurlindenCoV, "N_VNew_Serial", 0)) return(1);
      logPosterior_can_block  = NULL; logPosterior_can_block = N_VNew_Serial(blocksPerRank); if (check_retval((void *)logPosterior_can_block, "N_VNew_Serial", 0)) return(1);
      logPosterior_curr_block = NULL; logPosterior_curr_block = N_VNew_Serial(blocksPerRank); if (check_retval((void *)logPosterior_curr_block, "N_VNew_Serial", 0)) return(1);
      LikelihoodNumerator = NULL; LikelihoodNumerator = N_VNew_Serial(blocksPerRank); if (check_retval((void *)LikelihoodNumerator, "N_VNew_Serial", 0)) return(1);
    }
    if (check_retval((void *)thetaCurr, "N_VNew_Serial", 0)) return(1);
    if (check_retval((void *)thetaCan, "N_VNew_Serial", 0)) return(1);
    // if (check_retval((void *)thetaCan_temp, "N_VNew_Serial", 0)) return(1);
    if(check_retval((void *)nAccepted, "N_VNew_Serial", 0)) return(1);

    if (chain_rank!=masterrank) {
      // preallocate vector to save the ode odeSolution
      odeSolution = NULL;

      // preallocate variables for sundials solver
      y_Sundials = abstol = NULL;
      A = NULL;
      LS = NULL;
      cvode_mem = NULL;
    }

    if (chain_rank==masterrank) {
      // create an instance of a random number generator of type gsl_rng_sprng20
      // gslrng = gsl_rng_alloc(gsl_rng_sprng20);
      gslrng = gsl_rng_alloc(gsl_rng_taus);
      if (seedBool!=0) {
        seed    = getSeed();            // get a seed based on current time. default seed = 0.
        gsl_rng_set(gslrng, seed);      // set a different seed for the rng.
      }
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
    /* nehme zum Testen die Mittelwerte aus Tabelle 6, S. 275.
    * d.h. ich bin schon im log-transformierten Bereich.
    */
    if (chain_rank==masterrank) {
      Ith(thetaCurr,1) = log(0.4); // sigma der Likelihood-Funktion
    } else {
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

      for (ctr = 0; ctr < blocksPerRank; ctr++) {
        kk=1;
        // Startwerte ein wenig auslenken: ein sigma weit weg (man könnte noch ein random * (+-1) einbauen, damit die Startwerte random links oder rechts vom Zielwert sind.)
        Ith(thetaCurr,ctr*NPARSTUDY+1) = log(0.332) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // hour (h)
        Ith(thetaCurr,ctr*NPARSTUDY+2) = log(0.0476) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // hour (h)
        Ith(thetaCurr,ctr*NPARSTUDY+3) = log(123.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+4) = log(2.57) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;     // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
        Ith(thetaCurr,ctr*NPARSTUDY+5) = log(1.2e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+6) = log(478.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+7) = log(0.345) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // unitless
        Ith(thetaCurr,ctr*NPARSTUDY+8) = log(467.0) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
        Ith(thetaCurr,ctr*NPARSTUDY+9) = log(6.14e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+10) = log(4.99e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+11) = log(0.343) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // unitless
        Ith(thetaCurr,ctr*NPARSTUDY+12) = log(5.21e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
        Ith(thetaCurr,ctr*NPARSTUDY+13) = log(1.75e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+14) = log(3.54e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // micromol per hour (\mymol/h)
        Ith(thetaCurr,ctr*NPARSTUDY+15) = log(2.23e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // microM (\myM)
        Ith(thetaCurr,ctr*NPARSTUDY+16) = log(1.4e+7) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // micromol per hour (\mymol/h)
        Ith(thetaCurr,ctr*NPARSTUDY+17) = log(3.6e+4) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // per hour (1/h)
        Ith(thetaCurr,ctr*NPARSTUDY+18) = log(3.66e+3) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;  // per hour (1/h)
        Ith(thetaCurr,ctr*NPARSTUDY+19) = log(0.0123) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;   // liter per hour per BW^{0.75} (L/h/BW^{0.75})
        Ith(thetaCurr,ctr*NPARSTUDY+20) = log(0.155) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
        Ith(thetaCurr,ctr*NPARSTUDY+21) = log(0.138) + log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk); kk++;    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
        // printf("para1=%g\n", Ith(thetaCurr,ctr*NPARSTUDY+1));
          // Ith(thetaCurr,ctr*NPARSTUDY+1) = log(0.332);    // hour (h)
          // Ith(thetaCurr,ctr*NPARSTUDY+2) = log(0.0476);   // hour (h)
          // Ith(thetaCurr,ctr*NPARSTUDY+3) = log(123.0);    // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+4) = log(2.57);     // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
          // Ith(thetaCurr,ctr*NPARSTUDY+5) = log(1.2e+3);   // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+6) = log(478.0);    // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+7) = log(0.345);    // unitless
          // Ith(thetaCurr,ctr*NPARSTUDY+8) = log(467.0);    // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
          // Ith(thetaCurr,ctr*NPARSTUDY+9) = log(6.14e+3);  // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+10) = log(4.99e+4);  // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+11) = log(0.343);    // unitless
          // Ith(thetaCurr,ctr*NPARSTUDY+12) = log(5.21e+3);  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
          // Ith(thetaCurr,ctr*NPARSTUDY+13) = log(1.75e+4);  // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+14) = log(3.54e+4);  // micromol per hour (\mymol/h)
          // Ith(thetaCurr,ctr*NPARSTUDY+15) = log(2.23e+4);  // microM (\myM)
          // Ith(thetaCurr,ctr*NPARSTUDY+16) = log(1.4e+7);   // micromol per hour (\mymol/h)
          // Ith(thetaCurr,ctr*NPARSTUDY+17) = log(3.6e+4);   // per hour (1/h)
          // Ith(thetaCurr,ctr*NPARSTUDY+18) = log(3.66e+3);  // per hour (1/h)
          // Ith(thetaCurr,ctr*NPARSTUDY+19) = log(0.0123);   // liter per hour per BW^{0.75} (L/h/BW^{0.75})
          // Ith(thetaCurr,ctr*NPARSTUDY+20) = log(0.155);    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
          // Ith(thetaCurr,ctr*NPARSTUDY+21) = log(0.138);    // liter per hour per BW^{0.75} (L/h/BW^{0.75})
      }

      /* Create serial vector of length NEQ for I.C. and abstol */
      y_Sundials = N_VNew_Serial(NEQ); if (check_retval((void *)y_Sundials, "N_VNew_Serial", 0)) return(1);
      abstol = N_VNew_Serial(NEQ); if (check_retval((void *)abstol, "N_VNew_Serial", 0)) return(1);
    }

    /* Create user's data */
    data = (UserData) malloc(sizeof *data); if(check_retval((void *)data, "malloc", 2)) return(1);

    /****************************************************************************/
    /************************** Vorbereitungen **********************************/
    /****************************************************************************/
    if (chain_rank==masterrank) {
      printf(" \nacetaminophen problem (Zurlinden 2015)\n\n");

      swapscheme = (int *) calloc(swaps_max, sizeof(int));

      // Speicherplatz für sigmaRW_send belegen:
      sigmaRW_send = (double *) malloc(((NPARSTUDY+1) * (NSTUDIES+1)) * sizeof(double)); // einmal mehr (NPARSTUDY+1), weil der master auch an selber sendet...
      sendcounts  = (int *) malloc(chain_size*sizeof(int));
      displs      = (int *) malloc(chain_size*sizeof(int));
      // Übergabeparameter für MPI_Iscatterv - masterrelevant:
      // sendcounts[0] = 0;
      // displs[0]     = 0;
      for (i = 0; i < chain_size; i++) {
        sendcounts[i] = blocksPerRank*(NPARSTUDY+1);
        displs[i]     = i*blocksPerRank*(NPARSTUDY+1);
        // printf("%d %d\n",sendcounts[i],displs[i]);
      }
      // sendcounts[0] = 0;
    }
    else{
      /************************* init SUNDIALS ************************************/
      /* Initialize y_Sundials. Muss unten bei reinit auch so stehen. */
      for (i = 1; i <= NEQ; i++) { Ith(y_Sundials,i) = 0.0; }
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

      /* Speicherplatz für swap-Variablen belegen */
      thetaCurr_swap_send = (double *) malloc(blocksPerRank*NPARSTUDY * sizeof(double));
      thetaCurr_swap_recv = (double *) malloc(blocksPerRank*NPARSTUDY * sizeof(double));
      logPosterior_curr_block_send = (double *) malloc(blocksPerRank * sizeof(double));
      logPosterior_curr_block_recv = (double *) malloc(blocksPerRank * sizeof(double));
    }

    // Übergabeparameter für MPI_Iscatterv - alle relevant:
    sigmaRW_recv  = (double *) malloc((blocksPerRank*(NPARSTUDY + 1)) * sizeof(double));
    recvcount     = blocksPerRank*NPARSTUDY + blocksPerRank;

    /****************************************************************************/
    /****************************************************************************/
    /***********************        main loop        ****************************/
    /****************************************************************************/
    /****************************************************************************/
    if (chain_rank==masterrank) {
      Ith(nAccepted,1) = 0;
    } else {
      for (kk = 1; kk <= blocksPerRank; kk++) {Ith(nAccepted,kk) = 0;}
    }
    acceptrate = 0.0; onepercent = iterAll/100; waitbar_counter = 0;
    starttime_iterAll = MPI_Wtime();  // stop time of main loop

    for (iterJ = 0; iterJ < iterAll; iterJ++) {

      if (chain_rank==masterrank) {
        /* waitbar */
        if (iterJ == waitbar_counter*onepercent) {
          printf(" world_rank %d: waitbar: %d%% ", world_rank, waitbar_counter*1);
          waitbar_counter++;
        }
      }
      fprintf(fileTimes, "\niter\t%d", iterJ);

      /**************************************************************************/
      /*************************** Block: alle Studien **************************/
      /**************************************************************************/
      if (chain_rank!=masterrank) {
        // initiate a broadcast relating to sigmaLL
        starttime_bcast_sigmaLL = MPI_Wtime();
        MPI_Bcast(&sigmaLL,1,MPI_DOUBLE,masterrank,chain_comm); // recv current sigmaLL for LikelihoodTheta function.
        endtime_bcast_sigmaLL = MPI_Wtime(); cputime_bcast_sigmaLL = endtime_bcast_sigmaLL - starttime_bcast_sigmaLL; fprintf(fileTimes, "\nposteriorBcastSigmaLL\t%d\t%10.6e", studyblock, cputime_bcast_sigmaLL);
        cputime_bcast_sigmaLL_sum += cputime_bcast_sigmaLL;
        // MPI_Irecv(&sigmaLL,1,MPI_DOUBLE,masterrank,tag,chain_comm,&request_bcast);
        ctr = 0; // für passenden Zugriff bei den Vektoren bzgl. des studyblocks

        for (studyblock = chain_rank * blocksPerRank; studyblock < (chain_rank+1)*blocksPerRank; studyblock++) {
          fprintf(fileCandidates, "%d\t%d", iterJ, studyblock);
          /************************************************************************/
          /************************** draw new candidate **************************/
          /************************************************************************/
          // random walk, i.e.: can = curr + randn * stdv;
          // ...corresponding to results of Zurlinden, i.e. use Zurlinden's posterior results as proposal distribution
          /* Krauss,2015,S.7: Each parameter of one block is sampled independently,
           *   such that the proposal covariance matrix becomes diagonal. --> d.h. ich
           *   kann jeden Parameter einzeln random-walken.
           */
          starttime_proposal = MPI_Wtime();
          if (iterJ>0) { // ab iterJ==1 muss auf jeden Fall gewährleistet sein, dass sigmaRW_recv angekommen ist.
            if (ctr==0) { // zunächst warten, sodass alles angekommen ist
              MPI_Wait(&request_sigmaRW,MPI_STATUS_IGNORE);
              // for (i = 0; i < blocksPerRank*(NPARSTUDY + 1); i++) {
              //   printf("c_%d %d %f\n",studyblock,i,sigmaRW_recv[i]);
              // }
            }
            // for (i = 0; i < ((NPARSTUDY+1)*(NSTUDIES+1)); i++) {
            //   printf("%d %f\n",i,sigmaRW_recv[i]);
            // }
            logaccrateUniform = sigmaRW_recv[ctr*(NPARSTUDY + 1)+NPARSTUDY]; // muss hierhin, damit buffer sicher ist.
            // printf("c_%d [%d*(%d+1)+%d]=%f\n",studyblock,ctr,NPARSTUDY,NPARSTUDY,logaccrateUniform);
          }
          for (kk = 1; kk <= NPARSTUDY; kk++) {
            Ith(thetaCan,kk) = Ith(thetaCurr,ctr*NPARSTUDY+kk);
            // if (iterJ>0) { // bei iterJ==0 den Startwertvektor als "Kandidatenvektor" nehmen, damit ich nicht alles doppelt schreiben muss.
            //   sigmaRW = sigmaRW_scale * log(Ith(posteriorZurlindenMean,kk)) * Ith(posteriorZurlindenCoV,kk);
            //   sigmaRW = 1e-1;
            //   Ith(thetaCan,kk) += gsl_ran_gaussian(gslrng,sigmaRW);
            // }
            if (iterJ>0) {
              Ith(thetaCan,kk) += sigmaRW_recv[ctr*(NPARSTUDY+1) + kk-1];
              // printf("c_%d %d=%f\n",studyblock,kk,sigmaRW_recv[ctr*(NPARSTUDY+1) + kk-1]);
            }
            // printf("Can_%d: %f\n",kk, Ith(thetaCurr,ctr*NPARSTUDY+kk));
            fprintf(fileCandidates, "\t%14.6e", Ith(thetaCan,kk));
          }
          // empfange vom Master die gewürfelten sigmaRW's. Einmal alle zusammen. schonmal bei iterJ==0 starten, um möglichst kein warten zu haben.
          if ((iterJ<iterAll-1)&&(studyblock==(chain_rank+1)*blocksPerRank-1)) {
            MPI_Iscatterv(sigmaRW_send, sendcounts, displs, MPI_DOUBLE, sigmaRW_recv, recvcount, MPI_DOUBLE, masterrank, chain_comm, &request_sigmaRW); // recv current sigmaLL for LikelihoodTheta function.
          }

          endtime_proposal = MPI_Wtime(); cputime_proposal = endtime_proposal - starttime_proposal; // fprintf(fileTimes, "\nproposal\t%d\t%10.6e", studyblock, cputime_proposal);
          /************************************************************************/
          /********************* posterior of candidate ***************************/
          /************************************************************************/
          // benötige: Kandidaten, Daten
          logLikelihood_can = 0.0;
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

          /* reset UserData data according to current iteration, i.e. to current thetaCan. */
          data->T_G           = exp(Ith(thetaCan,1));   // hour (h)
          data->T_P           = exp(Ith(thetaCan,2));   // hour (h)
          data->K_APAP_Mcyp   = exp(Ith(thetaCan,3));   // microM (\myM)
          data->V_MCcyp       = exp(Ith(thetaCan,4));   // micromol per hour and per BW^{0.75} (\mymol/h/BW^{0.75})
          data->K_APAP_Msult  = exp(Ith(thetaCan,5));   // microM (\myM)
          data->K_APAP_Isult  = exp(Ith(thetaCan,6));   // microM (\myM)
          data->K_PAPS_Msult  = exp(Ith(thetaCan,7));   // unitless
          data->V_MCsult      = exp(Ith(thetaCan,8));   // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
          data->K_APAP_Mugt   = exp(Ith(thetaCan,9));   // microM (\myM)
          data->K_APAP_Iugt   = exp(Ith(thetaCan,10));  // microM (\myM)
          data->K_UDPGA_Mugt  = exp(Ith(thetaCan,11));  // unitless
          data->V_MCugt       = exp(Ith(thetaCan,12));  // microM per hour and per BW^{0.75} (\myM/h/BW^{0.75})
          data->K_APAPG_Mmem  = exp(Ith(thetaCan,13));  // microM (\myM)
          data->V_APAPG_Mmem  = exp(Ith(thetaCan,14));  // micromol per hour (\mymol/h)
          data->K_APAPS_Mmem  = exp(Ith(thetaCan,15));  // microM (\myM)
          data->V_APAPS_Mmem  = exp(Ith(thetaCan,16));  // micromol per hour (\mymol/h)
          data->k_synUDPGA    = exp(Ith(thetaCan,17));  // per hour (1/h)
          data->k_synPAPS     = exp(Ith(thetaCan,18));  // per hour (1/h)
          data->k_APAP_R0     = exp(Ith(thetaCan,19));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})
          data->k_APAPG_R0    = exp(Ith(thetaCan,20));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})
          data->k_APAPS_R0    = exp(Ith(thetaCan,21));  // liter per hour per BW^{0.75} (L/h/BW^{0.75})
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

          // print inital conditions into file
          // fprintf(fileSolution, "iter %d\tblock %d\n", iterJ, studyblock);
          // fprintf(fileSolution,"t=%0.4e\t", 0.0);
          // for (i = 1; i <= 3; i++) fprintf(fileSolution,"%14.6e ", Ith(y_Sundials,i*10)); fprintf(fileSolution, "\n");
          endtime_ODE_reinit = MPI_Wtime(); cputime_ODE_reinit = endtime_ODE_reinit-endtime_ODE_IV; // fprintf(fileTimes, "\nposteriorReinitSundials\t%d\t%10.6e", studyblock, cputime_ODE_reinit);

          iout = 1;                 // variables for ode solver: iout counts the # of output times
          /* solve ode system */
          while(1) {
            tout    = Ith(data2fit,iout*(nAPAP_G_S+1)+1); // nächste output time setzen: Datenzeitpunkte sind an den Stellen 1, 1*(nAPAP_G_S+1)+1, 2*(nAPAP_G_S+1)+1, ...
            retval  = CVode(cvode_mem, tout, y_Sundials, &t, CV_NORMAL); // Löse bis zum Zeitpunkt tout

            if (check_retval(&retval, "CVode", 1)) {
              printf("iter %d rank %d studyblock %d\n", iterJ, world_rank, studyblock);
              fprintf(fileLog, "SUNDIALS_ERROR: world_rank %d -> CVode() failed with retval = %d: In iter = %d and studyblock = %d at t = %g from tout = %g\n\n", world_rank, retval, iterJ, studyblock, t, tout);
              endtime_ODE_fail = MPI_Wtime(); fprintf(fileTimes, "\nposteriorODEFailed\t%d\t%10.6e", studyblock, endtime_ODE_fail - endtime_ODE_reinit);
              cputime_ODE_fail += endtime_ODE_fail - endtime_ODE_reinit;
              ode_fail++;
              // fprintf(fileLog,"candidate_%3d_%2d: \n",iterJ,studyblock);
              // for (i = 1; i <= NPARSTUDY; i++) {
              //   fprintf(fileLog,"%14.6e ", Ith(thetaCan,i));
              // }
              // fprintf(fileLog,"\n");
              nAcceptBool = -2;
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
          /********************************************************************/
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

          /********************************************************************/
          /* likelihood berechnen (als LogLikelihood) */
          // MPI_Recv(&sigmaLL,1,MPI_DOUBLE,masterrank,tag,chain_comm,&status); // recv current sigmaLL for LikelihoodTheta function.
          // if (ctr==0) {
          //   MPI_Wait(&request_bcast, MPI_STATUS_IGNORE); // bzgl. MPI_Ibcast sigmaLL. Wurde vor der for-Schleife angestoßen.
          // }
          // endtime_bcast_sigmaLL = MPI_Wtime(); cputime_bcast_sigmaLL = endtime_bcast_sigmaLL - endtime_ODE; fprintf(fileTimes, "\nposteriorBcastSigmaLL\t%d\t%10.6e", studyblock, cputime_bcast_sigmaLL);
          // cputime_bcast_sigmaLL_sum += cputime_bcast_sigmaLL;

          retval = getLogLikelihoodTheta(nData, nAPAP_G_S, &logLikelihood_can_blockj, &LikelihoodNumerator_blockj_can, odeSolution, data2fit, exp(sigmaLL), LogNormalBool);
          logLikelihood_can = logLikelihood_can_blockj;
          if (studyblock==1) {
            // printf("ndata:%d nAPAP_G_S:%d logLikelihood_can_blockj:%f LikelihoodNumerator_blockj_can:%f odeSolution... data2fit... sigmaLL:%f LogNormalBool:%d\n", nData, nAPAP_G_S, logLikelihood_can_blockj, LikelihoodNumerator_blockj_can, exp(sigmaLL), LogNormalBool);
          }
          // MPI_Gather(); // send LikelihoodNumerator_blockj_can an masterrank - nur wenn thetaCan akzeptiert wird!

          /********************************************************************/
          /* prior der Kandidaten berechnen (als LogPrior) */
          retval = getLogPrior(&logPrior_can, thetaCan);                        // thetaCan sind im log-Bereich. Für Priorauswertung in normalen Bereich zurück? Dann müssten Priorverteilungen auch entsprechend für den normalen Bereich sein...

          // zähle mal die Likelihoods, die nan sind.
          if (isnan(logLikelihood_can)){ // ||(logLikelihood_can==-inf)) {
            logLikelihood_nan++;
          }

          /* posterior berechnen (LogPosterior) */
          Ith(logPosterior_can_block,(ctr+1)) = logLikelihood_can + logPrior_can;

          if (iterJ==0) {
            retval = printSolutionToFile(fileSolution, iterJ, studyblock, nData, data2fit, nAPAP_G_S, odeSolution);
          }

          endtime_posterior = MPI_Wtime();
          cputime_posterior_rest = endtime_posterior - endtime_gather_LLNumer; // fprintf(fileTimes, "\nposteriorRest\t%d\t%10.6e", studyblock, cputime_posterior_rest);
          cputime_posterior = endtime_posterior - endtime_proposal; fprintf(fileTimes, "\nposterior\t%d\t%10.6e", studyblock, cputime_posterior);

          if (iterJ>0) { // bei j==0 wird noch der theta_Anfangswertvektor ausgewertet
            /************************************************************************/
            /****************** akzeptanzrate berechnen *****************************/
            /************************************************************************/
            logaccrate = Ith(logPosterior_can_block,(ctr+1)) - Ith(logPosterior_curr_block,(ctr+1)); // in dieser Form, solange Proposal symmetrisch ist! Sonst nochmal ändern!

            /************************************************************************/
            /****************** accept-reject Schritt *******************************/
            /************************************************************************/
            // logaccrateUniform = log(gsl_rng_uniform(gslrng));             // LogUniform, um mit LogPosterior vergleichen zu können!
            // siehe oben nach MPI_Wait(&request_sigmaRW,...): logaccrateUniform = sigmaRW_recv[ctr*(NPARSTUDY + 1)+NPARSTUDY];

            if(logaccrateUniform < logaccrate) { // accept
              Ith(nAccepted,ctr+1) += 1;
              nAcceptBool = 1;

              fprintf(fileMarkovChain, "%d\t%d", iterJ, studyblock);
              for (kk = 1; kk <= NPARSTUDY; kk++) {
                // save candidate into current
                Ith(thetaCurr,ctr*NPARSTUDY+kk) = Ith(thetaCan,kk);
                // write thetaCurr to file
                fprintf(fileMarkovChain, "\t%14.6e", Ith(thetaCurr,ctr*NPARSTUDY+kk));
              }
              fprintf(fileMarkovChain, "\n");

              retval = printSolutionToFile(fileSolution, iterJ, studyblock, nData, data2fit, nAPAP_G_S, odeSolution);

              Ith(logPosterior_curr_block,(ctr+1)) = Ith(logPosterior_can_block,(ctr+1));
              Ith(LikelihoodNumerator,ctr+1) = LikelihoodNumerator_blockj_can;

            } else { // reject
              if (nAcceptBool!=-2) {                                            // ist -2, falls sundials einen Fehler hatte. Dann wird der Block (hoffentlich) nicht akzeptiert worden sein.
                nAcceptBool = 0;
              }
            }
            fprintf(fileAcceptReject, "%14.6e\t%14.6e\t%d\n",logaccrateUniform, logaccrate, nAcceptBool);
            endtime_accrej = MPI_Wtime(); cputime_accrej = endtime_accrej - endtime_posterior; // fprintf(fileTimes, "\naccrej\t%d\t%10.6e", studyblock, cputime_accrej);

          } else { // iterJ == 0
            fprintf(fileMarkovChain, "%d\t%d", iterJ, studyblock); for (kk = 1; kk <= NPARSTUDY; kk++) {fprintf(fileMarkovChain, "\t%14.6e", Ith(thetaCurr,ctr*NPARSTUDY+kk));} fprintf(fileMarkovChain, "\n");

            Ith(logPosterior_curr_block,(ctr+1)) = Ith(logPosterior_can_block,(ctr+1));
            Ith(LikelihoodNumerator,ctr+1) = LikelihoodNumerator_blockj_can;

            nAcceptBool = 1;
          }

          endtime_ODE = MPI_Wtime();
          // MPI_Gather(); // send LikelihoodNumerator_blockj an masterrank
          MPI_Send(&Ith(LikelihoodNumerator,ctr+1),1,MPI_DOUBLE,masterrank,tag,chain_comm); // sende aktuellen LikelihoodNumerator_blockj an masterrank
          endtime_gather_LLNumer = MPI_Wtime(); cputime_gather_LLNumer = endtime_gather_LLNumer - endtime_ODE; fprintf(fileTimes, "\nposteriorGatherLLNumer\t%d\t%10.6e", studyblock, cputime_gather_LLNumer);
          cputime_gather_LLNumer_sum += cputime_gather_LLNumer;

          fprintf(fileCandidates, "\t%d\n", nAcceptBool);
          nAcceptBool=0;                                                        // immer zurücksetzen, damit er in  nächster Iteration nicht mehr auf -2 steht.
          ctr++;                                                                // Zugriff auf nächsten Studienblock passend setzen
        } // for-loop studyblock
      } // if chain_rank!=masterrank
      else {
        /**************************************************************************/
        /**************************** Block: sigmaLL ******************************/
        /**************************************************************************/

        starttime_proposal = MPI_Wtime();
        /************************** draw new candidate **************************/
        // for the clients:
        if (iterJ>0) {
          MPI_Wait(&request_sigmaRW,MPI_STATUS_IGNORE);
          // for (i = 0; i < NPARSTUDY+1; i++) {
          //   printf("m_%i %f\n",i,sigmaRW_recv[i]);
          // }
        }

        if (iterJ>0) {
          // Zufallszahlen für sigmaLL + Acc/Rej: hier vor, damit der Zufallsgenerator in gleicher Reihenfolge wie bei blockMH arbeitet.
          sigmaLL_Can_next = gsl_ran_gaussian(gslrng,1e-3);
          sigmaLL_AccRej_next = log(gsl_rng_uniform(gslrng));
          fprintf(filenIterSundials, "%4d %2d  0 %f\n",iterJ,NSTUDIES+1,sigmaLL_Can_next);
          fprintf(filenIterSundials, "%4d %2d  0 %f\n",iterJ,NSTUDIES+1,sigmaLL_AccRej_next);

          // swapscheme bereits hier würfeln, damit die Zufallszahlen alle in derselben Reihenfolge gewürfelt werden.
          if ((swaps>1) && (nChains>1)) { // tausche nur auf Befehl && ansonsten gibt es keine Tauschpartner
            if (chain_rank==masterrank) { // beachte nur die Kettenmaster
              if (world_rank==masterrank) { // eine Kette würfelt das Schema

                retval = getSwapscheme(swaps_max, swaps, gslrng, swapscheme);
              }
            }
          }
        }

        for (i = 0; i < NSTUDIES; i++) {
          // Zufallszahlen für Studien: Parameter + Acc/Rej
          for (k = 0; k < NPARSTUDY; k++) {
            param_Can = gsl_ran_gaussian(gslrng,1e-1); // schon bei iterJ==0 anfangen, damit die dann per MPI_Iscatterv möglichst nichtblockierend versendet werden
            sigmaRW_send[i*(NPARSTUDY+1) + k] = param_Can;
            fprintf(filenIterSundials, "%4d %2d %2d %f\n",iterJ, i, k, param_Can);
            // if (i==0) {
            //   sigmaRW_send[NSTUDIES*(NPARSTUDY+1)+k] = sigmaRW_send[i*(NPARSTUDY+1) + k];
            // }
          }
          param_Acc = log(gsl_rng_uniform(gslrng));
          sigmaRW_send[i*(NPARSTUDY+1) + NPARSTUDY] = param_Acc;
          fprintf(filenIterSundials, "%4d %2d %2d %f\n",iterJ, i, NPARSTUDY+1, param_Acc);
          // if (i==0) {
          //   sigmaRW_send[NSTUDIES*(NPARSTUDY+1)+NPARSTUDY] = sigmaRW_send[i*(NPARSTUDY+1) + NPARSTUDY];
          // }
        }
        // for (i = 0; i < ((NPARSTUDY+1)*(NSTUDIES+1)); i++) {
        //   printf("%d %f\n",i,sigmaRW_send[i]);
        // }
        if (iterJ<iterAll-1) {
          // MPI_Iscatterv(const void* sendbuf, const int sendcounts[],
          //      const int displs[], MPI_Datatype sendtype, void* recvbuf,
          //      int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
          //      MPI_Request *request)
          MPI_Iscatterv(sigmaRW_send, sendcounts, displs, MPI_DOUBLE, MPI_IN_PLACE, recvcount, MPI_DOUBLE, masterrank, chain_comm, &request_sigmaRW);
        }

        // for sigmaLL:
        Ith(thetaCan,1) = Ith(thetaCurr,1);
        // bcast an alle clients. Hier, damit sigmaLL in derselben Reihenfolge wie bei blockMH verwendet wird.
        sigmaLL = Ith(thetaCan,1);
        MPI_Bcast(&sigmaLL,1,MPI_DOUBLE,masterrank,chain_comm); // initiate a broadcast relating to sigmaLL
        endtime_bcast_sigmaLL = MPI_Wtime(); cputime_bcast_sigmaLL = endtime_bcast_sigmaLL - starttime_proposal; fprintf(fileTimes, "\nposteriorBcastSigmaLL\t%d\t%10.6e", studyblock, cputime_bcast_sigmaLL);
        cputime_bcast_sigmaLL_sum += cputime_bcast_sigmaLL;

        if (iterJ>0) {
          // sigmaLL würfeln
          // Ith(thetaCan,1) = 0.5+gsl_rng_uniform(gslrng);
          // Ith(thetaCan,1) += gsl_ran_gaussian(gslrng,1e-3); // würfele sigmaLL mittels Random Walk
          Ith(thetaCan,1) += sigmaLL_Can_next; // würfele sigmaLL mittels Random Walk
        }
        fprintf(fileCandidates, "%d\t%d\t%14.6e", iterJ, 10, Ith(thetaCan,1));

        endtime_proposal = MPI_Wtime(); cputime_proposal = endtime_proposal - endtime_bcast_sigmaLL; studyblock=RANKMASTER; // fprintf(fileTimes, "\nproposal\t%d\t%10.6e", studyblock, cputime_proposal);

        /**************************************************************************/
        /********************* posterior of candidate *****************************/
        /**************************************************************************/
        /* Likelihood: Daten, ODE-Lösung, sigmaLL */
        // bcast sigmaLL
        // if (iterJ>0) { // erst ab zweiter Iteration anhalten, damit möglichst lange nicht blockiert wird.
        //   MPI_Wait(&request_bcast, MPI_STATUS_IGNORE); // bzgl. MPI_Ibcast sigmaLL
        // }
        // Jetzt Zeit messen und mit endtime_proposal vergleich. Dann erhalte ich die Zeit, die Ibcast noch zusätzlich benötigt.
        // endtime_bcast_sigmaLL = MPI_Wtime(); cputime_bcast_sigmaLL = endtime_bcast_sigmaLL - endtime_proposal; fprintf(fileTimes, "\nposteriorBcastSigmaLL\t%d\t%10.6e", studyblock, cputime_bcast_sigmaLL);
        // cputime_bcast_sigmaLL_sum += cputime_bcast_sigmaLL;


        logLikelihood_can = 0.0;
        // MPI_Gather(); // recv LikelihoodNumerator_blockj an masterrank
        for (i = 0; i < nClients; i++) { // nClients
          for (studyblock = i * blocksPerRank; studyblock < (i+1)*blocksPerRank; studyblock++) {
            starttime_ODE = MPI_Wtime();
            // Messdaten:
            retval = getDataInfos(studyblock, &nData, &nAPAP_G_S, data); //liefert passend zum block die passenden Infos: nData = # Messdatenpunkten; nAPAP_G_S = wie viele von APAP und/oder APAP-G und/oder APAP-S sind in den Daten
            // create vector for fitting data of current studyblock
            data2fit = N_VNew_Serial(nData*(nAPAP_G_S+1)); if (check_retval((void *)data2fit, "N_VNew_Serial", 0)) return(1); // +1 weil zusätzlich der Zeitpunkt der Daten gespeichert wird
            // data2fit mit passenden Werten füllen
            // retval = getData(studyblock, data2fit); // liefert passend zum studyblock die Daten, an die gefittet werden soll.
            retval = getDataSynthetic(studyblock, data2fit);

            endtime_getInfos = MPI_Wtime(); cputime_getInfos = endtime_getInfos - starttime_ODE; // fprintf(fileTimes, "\nposteriorGetInfos\t%d\t%10.6e", studyblock, cputime_getInfos);

            /************************* löse ODE System ****************************/
            /* likelihood berechnen (als LogLikelihood) */
            MPI_Recv(&LikelihoodNumerator_blockj_curr,1,MPI_DOUBLE,i,tag,chain_comm,&status);
            endtime_gather_LLNumer = MPI_Wtime(); cputime_gather_LLNumer = endtime_gather_LLNumer - endtime_getInfos; fprintf(fileTimes, "\nposteriorGatherLLNumer\t%d\t%10.6e", studyblock, cputime_gather_LLNumer);
            cputime_gather_LLNumer_sum += cputime_gather_LLNumer;

            Ith(LikelihoodNumerator,studyblock+1) = LikelihoodNumerator_blockj_curr;

            retval = getLogLikelihoodSigma(nData, nAPAP_G_S, &logLikelihood_can_blockj, LikelihoodNumerator, studyblock+1, data2fit, exp(Ith(thetaCan,1)), LogNormalBool);

            logLikelihood_can += logLikelihood_can_blockj;

            endtime_ODE = MPI_Wtime(); cputime_ODE_solve = endtime_ODE - endtime_gather_LLNumer; fprintf(fileTimes, "\nposteriorgetLogLi\t%d\t%10.6e", studyblock, cputime_ODE_solve);
            if (iterJ>0) {
              cputime_ODE_sum += cputime_ODE_solve;
            }
          }
        }

        if (isnan(logLikelihood_can)) {
          logLikelihood_nan++;
        }

        /* Prior von sigmaLL */
        // logPrior_can -= 2.0*log(exp(Ith(thetaCan,1))); // Jeffrey: 1/sigma^2 .
        logPrior_can = - 0.5*log(2*M_PI) - log(1*exp(Ith(thetaCan,1))) - pow(Ith(thetaCan,1) +1,2) / (2*1); // log( logNormal(-1,1) )

        /* posterior berechnen (LogPosterior) */
        logPosterior_can_sigma = logLikelihood_can + logPrior_can;

        endtime_posterior = MPI_Wtime();
        cputime_posterior_rest = endtime_posterior - endtime_ODE; studyblock = RANKMASTER; // fprintf(fileTimes, "\nposteriorRest\t%d\t%10.6e", studyblock, cputime_posterior_rest);
        cputime_posterior = endtime_posterior - endtime_proposal; fprintf(fileTimes, "\nposterior\t%d\t%10.6e", studyblock, cputime_posterior);

        if (iterJ>0) { // bei j==0 wird noch der theta_Anfangswertvektor ausgewertet
          /************************************************************************/
          /****************** akzeptanzrate berechnen *****************************/
          /************************************************************************/
          logaccrate = logPosterior_can_sigma - logPosterior_curr_sigma; // in dieser Form, solange Proposal symmetrisch ist! Sonst nochmal ändern!

          /************************************************************************/
          /****************** accept-reject Schritt *******************************/
          /************************************************************************/
          // logaccrateUniform = log(gsl_rng_uniform(gslrng));             // LogUniform, um mit LogPosterior vergleichen zu können!
          logaccrateUniform = sigmaLL_AccRej_next;

          if(logaccrateUniform < logaccrate) { // accept
            Ith(nAccepted,1) += 1;
            nAcceptBool = 1;

            // save candidate into current
            Ith(thetaCurr,1) = Ith(thetaCan,1);
            // write thetaCurr to file
            fprintf(fileMarkovChain, "%d\t%d\t%14.6e\n", iterJ, 10, Ith(thetaCurr,1));

            logPosterior_curr_sigma = logPosterior_can_sigma;

          } else { // reject
            nAcceptBool = 0;
          }
          fprintf(fileAcceptReject, "%14.6e\t%14.6e\t%d\n",logaccrateUniform, logaccrate, nAcceptBool);
          endtime_accrej = MPI_Wtime(); cputime_accrej = endtime_accrej - endtime_posterior; // fprintf(fileTimes, "\naccrej\t%d\t%10.6e", studyblock, cputime_accrej);

        } else { // iterJ == 0
          fprintf(fileMarkovChain, "%d\t%d\t%14.6e\n", iterJ, 10, Ith(thetaCurr,1));
          logPosterior_curr_sigma = logPosterior_can_sigma;
          nAcceptBool = 1;
        }
        fprintf(fileCandidates, "\t%d\n", nAcceptBool);
        /*************************** Block sigmaLL ENDE ***************************/
      }
      if (iterJ>0) {
        /**********************************************************************/
        /**************************** swap chains *****************************/
        /**********************************************************************/
        /*
         * Idee:
         * Würfele auf dem ersten Masterrank das Tauschschema. Sende an jeden
         * Kettenmaster den Tauschpartner der Kette.
         * Jeder Kettenmaster sendet die Partnerkette an seine Clients.
         * Die Clients tauschen direkt miteinander.
         */
        // reset cputimes to 0.0
        cputime_swap_toss=0.0; cputime_swap_bcast=0.0; cputime_swap_barrier=0.0;
        cputime_swap_partner=0.0; cputime_swap_wait=0.0;

        starttime_swap = MPI_Wtime();
        if ((swaps>1) && (nChains>1)) { // tausche nur auf Befehl && ansonsten gibt es keine Tauschpartner
          if (chain_rank==masterrank) { // beachte nur die Kettenmaster
            if (world_rank==masterrank) { // eine Kette würfelt das Schema

              // retval = getSwapscheme(swaps_max, swaps, gslrng, swapscheme); // nach oben verschoben, damit die ZV in derselben RHfolge gezogen werden.

              // sende jeder (außer eigenen) Kette ihren Tauschpartner
              for (i = 0; i < nChains; i++) {
                chainmaster = i*chain_size+masterrank;
                if (chainmaster != world_rank) {
                  MPI_Send(&swapscheme[i], 1, MPI_INT, chainmaster, tag, MPI_COMM_WORLD);
                } else {
                  swapPartner = swapscheme[i];
                }
              }
            } else { // if world_rank==masterrank
              // Jeder Kettenmaster empfängt swapPartner
              MPI_Recv(&swapPartner, 1, MPI_INT, masterrank, tag, MPI_COMM_WORLD, &status);
            }

            // endtime_swap_bcast_master = MPI_Wtime(); cputime_swap_bcast_master = endtime_swap_bcast_master - endtime_accrej; fprintf(fileTimes, "\nswapbcastmasters\t%10.6e", cputime_swap_bcast_master);
            // und sendet diesen an alle Kettenclients weiter, falls world_chainnumber!=swapPartner
            for (i = 0; i < chain_size; i++) {
              if (i!=masterrank) { // chainmaster kennt swapPartner bereits
                MPI_Send(&swapPartner, 1, MPI_INT, i, tag, chain_comm);
              }
            }
          } else { // if chain_rank==masterrank
            // bin jetzt bei allen Kettenclients: empfange vom Kettenmaster den swapPartner
            MPI_Recv(&swapPartner, 1, MPI_INT, masterrank, tag, chain_comm, &status);
          }

          endtime_swap_bcast = MPI_Wtime(); cputime_swap_bcast = endtime_swap_bcast-endtime_accrej; fprintf(fileTimes, "\nswapbcast\t%10.6e", cputime_swap_bcast);
          MPI_Barrier(chain_comm);
          endtime_swap_barrier = MPI_Wtime(); cputime_swap_barrier = endtime_swap_barrier-endtime_swap_bcast; fprintf(fileTimes, "\nswapbarrier\t%10.6e", cputime_swap_barrier);

          /********************** perform the swap ****************************/
          if (world_chainnumber!=swapPartner) {
            // berechne den world_rank vom swapPartner-Block
            swapPartner = swapPartner*chain_size+chain_rank;
            // jeder Block sendet an seinen Partnerblock und empfängt von diesem
            // world_chainmaster haben nur sigmaLL zu tauschen, alle andern das ganze thetaCurr
            if (chain_rank==masterrank) {
              // printf("WR:%d CR:%d\n", world_rank, chain_rank);
              sigmaLL_send = Ith(thetaCurr,1);

              MPI_Isend(&sigmaLL_send, 1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &request_swap_theta);
              MPI_Isend(&logPosterior_curr_sigma, 1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &request_swap_logPosterior);
              MPI_Recv(&sigmaLL_recv, 1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &status_swap_theta);
              MPI_Recv(&logPosterior_swap_sigma, 1, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &status_swap_logPosterior);
            } else {
              for (i = 1; i <= blocksPerRank*NPARSTUDY; i++) {
                thetaCurr_swap_send[i-1] = Ith(thetaCurr,i);
                thetaCurr_swap_recv[i-1] = 0.0;
                if (i<=blocksPerRank) {
                  logPosterior_curr_block_send[i-1] = Ith(logPosterior_curr_block,i);
                  logPosterior_curr_block_recv[i-1] = 0.0;
                }
              }
              MPI_Isend(thetaCurr_swap_send, blocksPerRank*NPARSTUDY, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &request_swap_theta);
              MPI_Isend(logPosterior_curr_block_send, blocksPerRank, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &request_swap_logPosterior);
              MPI_Recv(thetaCurr_swap_recv, blocksPerRank*NPARSTUDY, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &status_swap_theta);
              MPI_Recv(logPosterior_curr_block_recv, blocksPerRank, MPI_DOUBLE, swapPartner, tag, MPI_COMM_WORLD, &status_swap_logPosterior);
            }
            endtime_swap_partner = MPI_Wtime(); cputime_swap_partner = endtime_swap_partner-endtime_swap_barrier; fprintf(fileTimes, "\nswappartner\t%10.6e", cputime_swap_partner);

            MPI_Wait(&request_swap_theta, &status_swap_theta);
            MPI_Wait(&request_swap_logPosterior, &status_swap_logPosterior);
            endtime_swap_wait = MPI_Wtime(); cputime_swap_wait = endtime_swap_wait-endtime_swap_partner; fprintf(fileTimes, "\nswapwait\t%10.6e", cputime_swap_wait);

            // speichere empfangene Daten in entsprechenden Variablen
            if (chain_rank==masterrank) {
              Ith(thetaCurr,1) = sigmaLL_recv;
              logPosterior_curr_sigma = logPosterior_swap_sigma;
            } else {
              for (i = 1; i <= blocksPerRank*NPARSTUDY; i++) {
                Ith(thetaCurr,i) = thetaCurr_swap_recv[i-1];
                if (i <= blocksPerRank) {
                  Ith(logPosterior_curr_block,i) = logPosterior_curr_block_recv[i-1];
                }
              }
            }
          } // if world_chainnumber!=swapPartner

          starttime_swap_barrier = MPI_Wtime();
          MPI_Barrier(chain_comm);
          endtime_swap_barrier = MPI_Wtime(); cputime_swap_barrier = endtime_swap_barrier - starttime_swap_barrier; fprintf(fileTimes, "\nswapBarrierEnd\t%10.6e", cputime_swap_barrier);

        } // if swaps > 1 && nChains > 1
        endtime_swap = MPI_Wtime(); fprintf(fileTimes, "\nswap\t%10.6e", endtime_swap - starttime_swap);
        cputime_swap += endtime_swap - starttime_swap;
      } // if iterJ>0

      fprintf(filenIterSundials, "\n");
    } // end of 'iterAll'
    printf("100%%\n");

    endtime_iterAll = MPI_Wtime();
    cputime_iterAll = endtime_iterAll - starttime_iterAll;

    fprintf(fileTimes, "\ncputimeIterAll\t%10.6e", cputime_iterAll);
    fprintf(filenIterSundials, "%d\t%10.6e\t%ld\t%10.6e\t%ld\t%10.6e\t%10.6e\n", seedBool, cputime_iterAll, nIterSundials_sum, (double) cputime_ODE_sum/nIterSundials_sum, nIterSundials_fail, (double) cputime_ODE_fail/nIterSundials_fail, cputime_bcast_sigmaLL_sum);

    /**************************************************************************/
    /********************** end of main loop **********************************/
    /**************************************************************************/

    // fprintf(fileLog, "world_rank_%d:\titerAll:\t%d\tcputime_iterAll:\t%g\n", world_rank, iterAll, cputime_iterAll);
    // if (chain_rank != masterrank) {
    //   fprintf(fileLog, "world_rank_%d:\tcputime_ODE_sum\t%g\t_mean:\t%g\n", world_rank, cputime_ODE_sum, (double) cputime_ODE_sum/iterAll/blocksPerRank);
    // }
    // fprintf(fileLog, "world_rank_%d:\tcputime_ODE_fail:\t%g\tmean:\t%g\t#ode fail:\t%d\n", world_rank, cputime_ODE_fail, (double) cputime_ODE_fail/(ode_fail), ode_fail);
    // fprintf(fileLog, "world_rank_%d:\tcputime_iterAll-ODE_fail:\t%g\n", world_rank, cputime_iterAll-cputime_ODE_fail);
    // fprintf(fileLog, "world_rank_%d:\tcputime_swap:\t%g\t_mean:\t%g\n", world_rank, cputime_swap, cputime_swap/iterAll);
    // fprintf(fileLog, "world_rank_%d:\tlogLikelihood_nan:\t%d\n", world_rank, logLikelihood_nan);
    // if (chain_rank!=masterrank) {
    //   for (i = 1; i <= blocksPerRank; i++) {
    //     acceptrate += Ith(nAccepted,i);
    //     fprintf(fileLog, "world_rank_%d:\taccepted_block%d:\t%d\trate:\t%g\n", world_rank, world_rank, (int) Ith(nAccepted,i), (double) Ith(nAccepted,i)/iterAll);
    //   }
    //   fprintf(fileLog, "world_rank_%d:\tnIter_Sundials_sum:\t%ld\t_mean:\t%g\n", world_rank, nIterSundials_sum, (double) cputime_ODE_sum/nIterSundials_sum);
    // } else {
    //   acceptrate = (double) Ith(nAccepted,1)/iterAll;
    //   fprintf(fileLog, "world_rank_%d:\taccepted\t%d\taccrate:\t%g\n", world_rank, (int) Ith(nAccepted,1), acceptrate);
    // }
    // fprintf(fileLog, "world_rank_%d:\tcputime_Comm:\t%g\tibcast:\t%g\tgather:\t%g\n", world_rank, cputime_bcast_sigmaLL_sum+cputime_gather_LLNumer_sum, cputime_bcast_sigmaLL_sum, cputime_gather_LLNumer_sum);


    fprintf(fileLog, "world_rank\t%d\naccepted:", world_rank);
    if (chain_rank!=masterrank) {
      for (kk = 1; kk <= blocksPerRank; kk++) {
        acceptrate += Ith(nAccepted,kk);
        fprintf(fileLog,"\t%d", kk);
      }
      fprintf(fileLog, "\n\t");
      for (kk = 1; kk <= blocksPerRank; kk++) {
        acceptrate += Ith(nAccepted,kk);
        fprintf(fileLog,"\t%d", (int)Ith(nAccepted,kk));
      }
      fprintf(fileLog, "\n\t");
      for (kk = 1; kk <= blocksPerRank; kk++) {
        acceptrate += Ith(nAccepted,kk);
        fprintf(fileLog,"\t%g", Ith(nAccepted,kk)/iterAll);
      }
      fprintf(fileLog, "\n");
      acceptrate = (double) acceptrate/iterAll/NSTUDIES;
    } else {
      fprintf(fileLog, "\t%d\n", (int)Ith(nAccepted,1));
      acceptrate = (double)Ith(nAccepted,1)/iterAll;
    }
    fprintf(fileLog, "accrate\t\t%g\n", acceptrate);
    fprintf(fileLog, "Zeiten\t\tZeiten\t\tZeiten\t\tZeiten\t\tZeiten\t\t\t\t\t\tnIter\n");
    fprintf(fileLog, "\t\tODE\t\t\t\t\t\tKommunikation\t\t\t\t\tSundials\n");
    fprintf(fileLog, "overall\t\tsum\t\tfail\t\tsum-fail\tbcast\t\tgather\t\tswap\t\tall\t\tfail\tsum-fail\n");
    fprintf(fileLog, "%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%10.4e\t%ld\t%ld\t%ld\n", cputime_iterAll, cputime_ODE_sum, cputime_ODE_fail, cputime_ODE_sum-cputime_ODE_fail, cputime_bcast_sigmaLL_sum, cputime_gather_LLNumer_sum, cputime_swap, nIterSundials_sum, nIterSundials_fail, nIterSundials_sum-nIterSundials_fail);

    if (chain_rank!=masterrank) {
      /* Free y_Sundials and abstol vectors */
      N_VDestroy(y_Sundials);
      N_VDestroy(abstol);

      /* Free integrator memory */
      CVodeFree(&cvode_mem);

      /* Free the linear solver memory */
      SUNLinSolFree(LS);

      /* Free the matrix memory */
      SUNMatDestroy(A);

      /* free malloc */
      free(thetaCurr_swap_send);
      free(thetaCurr_swap_recv);
      free(logPosterior_curr_block_send);
      free(logPosterior_curr_block_recv);
    }
    if (world_rank==masterrank) {
      free(swapscheme);
      free(sigmaRW_send);
      free(sendcounts);
      free(displs);
    }
    free(sigmaRW_recv);

    // close files
    fclose(fileSolution);
    fclose(fileMarkovChain);
    fclose(fileAcceptReject);

    MPI_Comm_free(&chain_comm);
  } // end of: if (color_split!=MPI_UNDEFINED)

  // close MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return(retval);
}
