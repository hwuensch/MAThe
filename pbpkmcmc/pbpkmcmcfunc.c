#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_rng.h>
//#include <gsl-sprng.h>
#include <gsl/gsl_randist.h>
//#include <mpi.h>

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

/* Type : UserData (contains some variables) */
typedef struct {
    realtype BW, D_oral, D_oral_rel;
    realtype T_G, T_P, K_APAP_Mcyp, V_MCcyp, K_APAP_Msult, K_APAP_Isult;
    realtype K_PAPS_Msult, V_MCsult, K_APAP_Mugt, K_APAP_Iugt, K_UDPGA_Mugt;
    realtype V_MCugt, K_APAPG_Mmem, V_APAPG_Mmem, K_APAPS_Mmem, V_APAPS_Mmem;
    realtype k_synUDPGA, k_synPAPS, k_APAP_R0, k_APAPG_R0, k_APAPS_R0;
} *UserData;

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */


int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    //realtype time;
    // Speicherplatz für aktuelle DGLn-Werte
    realtype C_APAP_fat, C_APAP_muscle, C_APAP_liver, C_APAP_kidney, A_APAP_GI, C_APAP_sp, C_APAP_rp, A_APAP_e, C_APAP_A, C_APAP_V, C_APAPG_fat, C_APAPG_muscle, C_APAPG_liver, A_APAPG_hep, C_APAPG_hep, C_APAPG_kidney, C_APAPG_sp, C_APAPG_rp, A_APAPG_e, C_APAPG_A, C_APAPG_V, C_APAPS_fat, C_APAPS_muscle, C_APAPS_liver, A_APAPS_hep, C_APAPS_hep, C_APAPS_kidney, C_APAPS_sp, C_APAPS_rp, A_APAPS_e, C_APAPS_A, C_APAPS_V, phi_PAPS_liver, phi_UDPGA_liver;
    // Speicherplatz für Gleichungen, Variablen im ODE-System
    realtype P_fatblood_APAP, P_muscleblood_APAP, P_liverblood_APAP, P_kidneyblood_APAP, P_spblood_APAP, P_rpblood_APAP, P_fatblood_APAPG, P_muscleblood_APAPG, P_liverblood_APAPG, P_kidneyblood_APAPG, P_spblood_APAPG, P_rpblood_APAPG, P_fatblood_APAPS, P_muscleblood_APAPS, P_liverblood_APAPS, P_kidneyblood_APAPS, P_spblood_APAPS, P_rpblood_APAPS;
    realtype Q_CC, Q_C, Q_fat, Q_muscle, Q_liver, Q_kidney, Q_sp, Q_rp;
    realtype V_fat, V_muscle, V_liver, V_kidney, V_sp, V_rp, V_BLA, V_BLV;
    realtype dAbsdt, M, InputDosis, D_IV;
    realtype v_cyp, v_APAPG, v_APAPS, v_APAPG_mem, v_APAPS_mem;
    realtype V_Msult, V_Mugt, k_APAP_R, k_APAPG_R, k_APAPS_R;
    // geschätzte Variablen
    realtype T_G, T_P, K_APAP_Mcyp, V_MCcyp, K_APAP_Msult, K_APAP_Isult, K_PAPS_Msult, V_MCsult, K_APAP_Mugt, K_APAP_Iugt, K_UDPGA_Mugt, V_MCugt, K_APAPG_Mmem, V_APAPG_Mmem, K_APAPS_Mmem, V_APAPS_Mmem, k_synUDPGA, k_synPAPS, k_APAP_R0, k_APAPG_R0, k_APAPS_R0;
    // Speicherplatz für Übergabewert user_data
    realtype BW, D_oral, D_oral_rel; // body weight, initiale oral dosis, marker if dosis is absolute or relative to body weight
    UserData data;

    data = (UserData) user_data;
    BW             = data->BW;
    D_oral         = data->D_oral;
    D_oral_rel     = data->D_oral_rel;
    T_G            = data->T_G;
    T_P            = data->T_P;
    K_APAP_Mcyp    = data->K_APAP_Mcyp;
    V_MCcyp        = data->V_MCcyp;
    K_APAP_Msult   = data->K_APAP_Msult;
    K_APAP_Isult   = data->K_APAP_Isult;
    K_PAPS_Msult   = data->K_PAPS_Msult;
    V_MCsult       = data->V_MCsult;
    K_APAP_Mugt    = data->K_APAP_Mugt;
    K_APAP_Iugt    = data->K_APAP_Iugt;
    K_UDPGA_Mugt   = data->K_UDPGA_Mugt;
    V_MCugt        = data->V_MCugt;
    K_APAPG_Mmem   = data->K_APAPG_Mmem;
    V_APAPG_Mmem   = data->V_APAPG_Mmem;
    K_APAPS_Mmem   = data->K_APAPS_Mmem;
    V_APAPS_Mmem   = data->V_APAPS_Mmem;
    k_synUDPGA     = data->k_synUDPGA;
    k_synPAPS      = data->k_synPAPS;
    k_APAP_R0      = data->k_APAP_R0;
    k_APAPG_R0     = data->k_APAPG_R0;
    k_APAPS_R0     = data->k_APAPS_R0;

    // printf("t: %g Bw: %g, T_G: %g\n", t, BW, T_G);

    /* bis hierhin fungiert D_oral_rel als boolean, ob D_oral die absolute Dosis
     bezeichnet (negativer Wert), oder ob D_oral die relative gespeichert hat.
     Ab danach fungiert es als Speicherplatz für die relative Dosis.
     */
    if (D_oral_rel == 0) { // d.h. absolute Dosis wird verwendet
        D_oral_rel = D_oral / BW; // relative Dosis (mg/kg)
    }
    else { // d.h. D_oral ist in relativer Menge
        D_oral_rel = D_oral;
        D_oral = D_oral * BW;
    }

    // blood_APAP flow rates (email Zurlinden 20.08.19, original taken from Brown et al. (1997)) for ODEs
    Q_CC     = 16.2;               // Cardiac output in liter/(hour*BW^{0.75})
    Q_C      = Q_CC*pow(BW,0.75);  // Cardiac output scaled by body weight
    Q_fat    = 0.052 * Q_C;        // other tissues are expressed as fractions of Q_C
    Q_muscle = 0.191 * Q_C; // directly taken from Brown et al. (1997)
    Q_liver  = 0.227 * Q_C; // directly taken from Brown et al. (1997)
    Q_kidney = 0.175 * Q_C; // directly taken from Brown et al. (1997)
    Q_sp     = 0.14 * Q_C;
    Q_rp     = 0.215 * Q_C; // not in Brown et al. (1997) -> in Zurlinden: 0.22, but with this the sum of all Q_T is not 1, so I adjusted Q_rp s.t. sum is 1.
    // volumes (email Zurlinden 20.08.19) expressed as liter for ODEs
    V_fat    = 0.214 * pow(BW,0.75);
    V_muscle = 0.4 * pow(BW,0.75);
    V_liver  = 0.0257 * pow(BW,0.75);
    V_kidney = 0.0044 * pow(BW,0.75);
    V_sp     = 0.185 * pow(BW,0.75);
    V_rp     = 0.0765 * pow(BW,0.75);
    V_BLA    = 0.0243 * pow(BW,0.75);
    V_BLV    = 0.0557 * pow(BW,0.75);

    // Gewebe-zu-Blut Verhältnis (table 3, p.271) für Gleichung (3)
    P_fatblood_APAP      = 0.447;
    P_muscleblood_APAP   = 0.687;
    P_liverblood_APAP    = 0.687;
    P_kidneyblood_APAP   = 0.711;
    P_spblood_APAP       = 0.606;
    P_rpblood_APAP       = 0.676;
    P_fatblood_APAPG     = 0.128;
    P_muscleblood_APAPG  = 0.336;
    P_liverblood_APAPG   = 0.321;
    P_kidneyblood_APAPG  = 0.392;
    P_spblood_APAPG      = 0.351;
    P_rpblood_APAPG      = 0.364;
    P_fatblood_APAPS     = 0.088;
    P_muscleblood_APAPS  = 0.199;
    P_liverblood_APAPS   = 0.203;
    P_kidneyblood_APAPS  = 0.261;
    P_spblood_APAPS      = 0.254;
    P_rpblood_APAPS      = 0.207;

    // I(t) in Gleichung (1)
    // "give" initial amount of drug dosis
    InputDosis = 0.0;
    // intravenöse Dosis
    D_IV = 0.0;

    // Kombination von Gleichung (1) für M und (2) für D_oral
    M = D_oral; /* printf("\t%14.6e\t%14.6e\n", DOSISORAL, M); */
    if (D_oral <= 1000) {// falls die absolute Dosis geringer als 1000mg ist.
        M = M*(0.0005*D_oral + 0.37);
    } else {
        M = M*0.88;
    }
    dAbsdt = M*(exp(-t/T_G) - exp(-t/T_P)) / (T_G - T_P);

    // DGLn: aktueller Zustand
    C_APAP_fat     = Ith(y,1);
    C_APAP_muscle  = Ith(y,2);
    C_APAP_liver   = Ith(y,3);
    C_APAP_kidney  = Ith(y,4);
    A_APAP_GI      = Ith(y,5);
    C_APAP_sp      = Ith(y,6);
    C_APAP_rp      = Ith(y,7);
    A_APAP_e       = Ith(y,8);
    C_APAP_A       = Ith(y,9); // im Paper ist der Index BLA: C_APAP_BLA     = Ith(y,9);
    C_APAP_V       = Ith(y,10);// im Paper ist der Index BLV: C_APAP_BLV     = Ith(y,10);
    C_APAPG_fat    = Ith(y,11);
    C_APAPG_muscle = Ith(y,12);
    C_APAPG_liver  = Ith(y,13);
    C_APAPG_hep    = Ith(y,14);
    C_APAPG_kidney = Ith(y,15);
    C_APAPG_sp     = Ith(y,16);
    C_APAPG_rp     = Ith(y,17);
    A_APAPG_e      = Ith(y,18);
    C_APAPG_A      = Ith(y,19);// im Paper ist der Index BLA: C_APAPG_BLA    = Ith(y,19);
    C_APAPG_V      = Ith(y,20);// im Paper ist der Index BLV: C_APAPG_BLV    = Ith(y,20);
    C_APAPS_fat    = Ith(y,21);
    C_APAPS_muscle = Ith(y,22);
    C_APAPS_liver  = Ith(y,23);
    C_APAPS_hep    = Ith(y,24);
    C_APAPS_kidney = Ith(y,25);
    C_APAPS_sp     = Ith(y,26);
    C_APAPS_rp     = Ith(y,27);
    A_APAPS_e      = Ith(y,28);
    C_APAPS_A      = Ith(y,29);// im Paper ist der Index BLA: C_APAPS_BLA    = Ith(y,29);
    C_APAPS_V      = Ith(y,30);// im Paper ist der Index BLV: C_APAPS_BLV    = Ith(y,30);
    phi_PAPS_liver = Ith(y,31);// Gleichung (6)
    phi_UDPGA_liver= Ith(y,32);// Gleichung (6)

    // Gleichungen (4), (5), (5), (7), (7), (8), (8) und (8)
    // (4)
    v_cyp        = (V_MCcyp*pow(BW,0.75) * C_APAP_liver) / (K_APAP_Mcyp + C_APAP_liver);
    // (5)
    v_APAPG      = (V_MCugt*pow(BW,0.75) * C_APAP_liver * phi_UDPGA_liver) / ((K_APAP_Mugt + C_APAP_liver + pow(C_APAP_liver,2) / K_APAP_Iugt) * (K_UDPGA_Mugt + phi_UDPGA_liver));
    v_APAPS      = (V_MCsult*pow(BW,0.75) * C_APAP_liver * phi_PAPS_liver) / ((K_APAP_Msult + C_APAP_liver + pow(C_APAP_liver,2) / K_APAP_Isult) * (K_PAPS_Msult + phi_PAPS_liver));
    // (7)
    v_APAPG_mem  = (V_APAPG_Mmem * C_APAPG_hep) / (K_APAPG_Mmem + C_APAPG_hep);
    v_APAPS_mem  = (V_APAPS_Mmem * C_APAPS_hep) / (K_APAPS_Mmem + C_APAPS_hep);
    // (8)
    k_APAP_R     = k_APAP_R0 * BW;
    k_APAPG_R    = k_APAPG_R0 * BW;
    k_APAPS_R    = k_APAPS_R0 * BW;

    // DGLn: nächster Zustand
    Ith(ydot,1)  = Q_fat * (C_APAP_A - (C_APAP_fat / P_fatblood_APAP)) / V_fat; // APAP Fett
    Ith(ydot,2)  = Q_muscle * (C_APAP_A - (C_APAP_muscle / P_muscleblood_APAP)) / V_muscle; // APAP Muskeln
    Ith(ydot,3)  = (dAbsdt + Q_liver * (C_APAP_A - (C_APAP_liver / P_liverblood_APAP)) - v_cyp - v_APAPG - v_APAPS) / V_liver; // APAP Leber
    Ith(ydot,4)  = (Q_kidney * (C_APAP_A - (C_APAP_kidney / P_kidneyblood_APAP)) - k_APAP_R * C_APAP_A) / V_kidney; // APAP Niere
    Ith(ydot,5)  = InputDosis - dAbsdt; // APAP Magen-Darm
    Ith(ydot,6)  = Q_sp * (C_APAP_A - (C_APAP_sp / P_spblood_APAP)) / V_sp; // APAP langsames Gewebe
    Ith(ydot,7)  = Q_rp * (C_APAP_A - (C_APAP_rp / P_rpblood_APAP)) / V_rp; // APAP schnelles Gewebe
    Ith(ydot,8)  = k_APAP_R * C_APAP_A; // APAP Urin
    Ith(ydot,9)  = Q_C * (C_APAP_V - C_APAP_A) / V_BLA; // APAP arterielles Blut
    Ith(ydot,10) = (Q_fat*(C_APAP_fat / P_fatblood_APAP) + Q_muscle*(C_APAP_muscle / P_muscleblood_APAP) + Q_liver*(C_APAP_liver / P_liverblood_APAP) + Q_kidney*(C_APAP_kidney / P_kidneyblood_APAP) + Q_sp*(C_APAP_sp / P_spblood_APAP) + Q_rp*(C_APAP_rp / P_rpblood_APAP) - Q_C*C_APAP_V + D_IV) / V_BLV;

    Ith(ydot,11) = Q_fat * (C_APAPG_A - (C_APAPG_fat / P_fatblood_APAPG)) / V_fat; // APAPG Fett
    Ith(ydot,12) = Q_muscle * (C_APAPG_A - (C_APAPG_muscle / P_muscleblood_APAPG)) / V_muscle; // APAPG Muskeln
    Ith(ydot,13) = (Q_liver * (C_APAPG_A - (C_APAPG_liver / P_liverblood_APAPG)) + v_APAPG_mem) / V_liver; // APAPG Leber
    Ith(ydot,14) = v_APAPG - v_APAPG_mem; // APAPG Hepatocyte
    Ith(ydot,15) = (Q_kidney * (C_APAPG_A - (C_APAPG_kidney / P_kidneyblood_APAPG)) - k_APAPG_R * C_APAPG_A) / V_kidney; // APAPG Niere
    Ith(ydot,16) = Q_sp * (C_APAPG_A - (C_APAPG_sp / P_spblood_APAPG)) / V_sp; // APAPG langsames Gewebe
    Ith(ydot,17) = Q_rp * (C_APAPG_A - (C_APAPG_rp / P_rpblood_APAPG)) / V_rp; // APAPG schnelles Gewebe
    Ith(ydot,18) = k_APAPG_R * C_APAPG_A; // APAPG Urin
    Ith(ydot,19) = Q_C * (C_APAPG_V - C_APAPG_A) / V_BLA; // APAPG arterielles Blut
    Ith(ydot,20) = (Q_fat*(C_APAPG_fat / P_fatblood_APAPG) + Q_muscle*(C_APAPG_muscle / P_muscleblood_APAPG) + Q_liver*(C_APAPG_liver / P_liverblood_APAPG) + Q_kidney*(C_APAPG_kidney / P_kidneyblood_APAPG) + Q_sp*(C_APAPG_sp / P_spblood_APAPG) + Q_rp*(C_APAPG_rp / P_rpblood_APAPG) - Q_C*C_APAPG_V) / V_BLV;

    Ith(ydot,21) = Q_fat * (C_APAPS_A - (C_APAPS_fat / P_fatblood_APAPS)) / V_fat; // APAPS Fett
    Ith(ydot,22) = Q_muscle * (C_APAPS_A - (C_APAPS_muscle / P_muscleblood_APAPS)) / V_muscle; // APAPS Muskeln
    Ith(ydot,23) = (Q_liver * (C_APAPS_A - (C_APAPS_liver / P_liverblood_APAPS)) + v_APAPS_mem) / V_liver; // APAPS Leber
    Ith(ydot,24) = v_APAPS - v_APAPS_mem; // APAPS Hepatocyte
    Ith(ydot,25) = (Q_kidney * (C_APAPS_A - (C_APAPS_kidney / P_kidneyblood_APAPS)) - k_APAPS_R * C_APAPS_A) / V_kidney; // APAPS Niere
    Ith(ydot,26) = Q_sp * (C_APAPS_A - (C_APAPS_sp / P_spblood_APAPS)) / V_sp; // APAPS langsames Gewebe
    Ith(ydot,27) = Q_rp * (C_APAPS_A - (C_APAPS_rp / P_rpblood_APAPS)) / V_rp; // APAPS schnelles Gewebe
    Ith(ydot,28) = k_APAPS_R * C_APAPS_A; // APAPS Urin
    Ith(ydot,29) = Q_C * (C_APAPS_V - C_APAPS_A) / V_BLA; // APAPS arterielles Blut
    Ith(ydot,30) = (Q_fat*(C_APAPS_fat / P_fatblood_APAPS) + Q_muscle*(C_APAPS_muscle / P_muscleblood_APAPS) + Q_liver*(C_APAPS_liver / P_liverblood_APAPS) + Q_kidney*(C_APAPS_kidney / P_kidneyblood_APAPS) + Q_sp*(C_APAPS_sp / P_spblood_APAPS) + Q_rp*(C_APAPS_rp / P_rpblood_APAPS) - Q_C*C_APAPS_V) / V_BLV;

    Ith(ydot,31) = -v_APAPS + k_synPAPS * (1 - phi_PAPS_liver); // fraction of cofactor PAPS available for metabolism
    Ith(ydot,32) = -v_APAPG + k_synUDPGA * (1 - phi_UDPGA_liver); // fraction of cofactor UDPGA available for metabolism

    return(0);
}

/*
 * g routine. Compute functions g_i(t,y) for i = 0,1.
 */

int g(realtype t, N_Vector y, realtype *gout, void *user_data)
{
    realtype y1, y3;

    y1 = Ith(y,1); y3 = Ith(y,3);
    gout[0] = y1 - RCONST(0.0001);
    gout[1] = y3 - RCONST(0.01);

    return(0);
}

/*
 * Jacobian routine. Compute J(t,y) = df/dy. *
 */

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    realtype y2, y3;

    y2 = Ith(y,2); y3 = Ith(y,3);

    IJth(J,1,1) = RCONST(-0.04);
    IJth(J,1,2) = RCONST(1.0e4)*y3;
    IJth(J,1,3) = RCONST(1.0e4)*y2;

    IJth(J,2,1) = RCONST(0.04);
    IJth(J,2,2) = RCONST(-1.0e4)*y3-RCONST(6.0e7)*y2;
    IJth(J,2,3) = RCONST(-1.0e4)*y2;

    IJth(J,3,1) = ZERO;
    IJth(J,3,2) = RCONST(6.0e7)*y2;
    IJth(J,3,3) = ZERO;

    return(0);
}

/*
 *------------------------------------------------------------------------------
 * Private helper functions
 *------------------------------------------------------------------------------
 */

void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3)
{
#if defined(SUNDIALS_EXTENDED_PRECISION)
    printf("At t = %0.4Le      y =%14.6Le  %14.6Le  %14.6Le\n", t, y1, y2, y3);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#else
    printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#endif

    return;
}

void PrintRootInfo(int root_f1, int root_f2)
{
    printf("    rootsfound[] = %3d %3d\n", root_f1, root_f2);

    return;
}

/*
 * Get and print some final statistics
 */

void PrintFinalStats(void *cvode_mem)
{
    long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
    int retval;

    retval = CVodeGetNumSteps(cvode_mem, &nst);
    check_retval(&retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
    check_retval(&retval, "CVodeGetNumRhsEvals", 1);
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
    retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
    check_retval(&retval, "CVodeGetNumErrTestFails", 1);
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

    retval = CVodeGetNumJacEvals(cvode_mem, &nje);
    check_retval(&retval, "CVodeGetNumJacEvals", 1);
    retval = CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
    check_retval(&retval, "CVodeGetNumLinRhsEvals", 1);

    retval = CVodeGetNumGEvals(cvode_mem, &nge);
    check_retval(&retval, "CVodeGetNumGEvals", 1);

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
           nst, nfe, nsetups, nfeLS, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
           nni, ncfn, netf, nge);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

int check_retval(void *returnvalue, const char *funcname, int opt)
{
    int *retval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && returnvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer ",
                funcname);
        return(1); }

    /* Check if retval < 0 */
    else if (opt == 1) {
        retval = (int *) returnvalue;
        if (*retval < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d ",
                    funcname, *retval);
            return(1); }}

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && returnvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer ",
                funcname);
        return(1); }

    return(0);
}

/* compare the solution at the final time 4e10s to a reference solution computed
 using a relative tolerance of 1e-8 and absolute tolerance of 1e-14 */
int check_ans(N_Vector y, realtype t, realtype rtol, N_Vector atol)
{
    int      passfail=0;        /* answer pass (0) or fail (1) retval */
    N_Vector ref;               /* reference solution vector        */
    N_Vector ewt;               /* error weight vector              */
    realtype err;               /* wrms error                       */
    realtype ONE=RCONST(1.0);

    /* create reference solution and error weight vectors */
    ref = N_VClone(y);
    ewt = N_VClone(y);

    /* set the reference solution data */
    NV_Ith_S(ref,0) = RCONST(5.2083495894337328e-08);
    NV_Ith_S(ref,1) = RCONST(2.0833399429795671e-13);
    NV_Ith_S(ref,2) = RCONST(9.9999994791629776e-01);

    /* compute the error weight vector, loosen atol */
    N_VAbs(ref, ewt);
    N_VLinearSum(rtol, ewt, RCONST(10.0), atol, ewt);
    if (N_VMin(ewt) <= ZERO) {
        fprintf(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
        return(-1);
    }
    N_VInv(ewt, ewt);

    /* compute the solution error */
    N_VLinearSum(ONE, y, -ONE, ref, ref);
    err = N_VWrmsNorm(ref, ewt);

    /* is the solution within the tolerances? */
    passfail = (err < ONE) ? 0 : 1;

    if (passfail) {
        fprintf(stdout, "\nSUNDIALS_WARNING: check_ans error=%g \n\n", err);
    }

    /* Free vectors */
    N_VDestroy(ref);
    N_VDestroy(ewt);

    return(passfail);
}

int printSolutionToFile(FILE *fileSolution, int iterJ, int studyblock, int nData, N_Vector data2fit, int nAPAP_G_S, N_Vector odeSolution) {
    int i, kk;
    // write inital venous concentrations into output file
    fprintf(fileSolution, "iter\t%d\tblock\t%d\n", iterJ, studyblock);
    for (i = 0; i < nData; i++) {
        fprintf(fileSolution,"%0.4e\t", Ith(data2fit, i*(nAPAP_G_S+1)+1));
        for (kk = 1; kk <= 3; kk++) {
            fprintf(fileSolution,"%14.10e\t", Ith(odeSolution,i*3 + kk));
        }
        fprintf(fileSolution, "\n");
    }
    return(0);
}
/******************************************************************************/
/*************************** functions for posterior **************************/
/******************************************************************************/

int getDataInfos(int rank, int *nData, int *nAPAP_G_S, void *user_data){
    UserData data;

    data = (UserData) user_data;

    if (rank==0)      { *nData = 14; *nAPAP_G_S = 3; data->BW = 73.11; data->D_oral = 80.0; data->D_oral_rel=1; }
    else if (rank==1) { *nData = 15; *nAPAP_G_S = 3; data->BW = 68.00; data->D_oral = 20.0; data->D_oral_rel=1; }
    else if (rank==2) { *nData = 15; *nAPAP_G_S = 3; data->BW = 57.00; data->D_oral = 80.0; data->D_oral_rel=1; }
    else if (rank==3) { *nData = 13; *nAPAP_G_S = 3; data->BW = 70.00; data->D_oral = 1000.0; data->D_oral_rel=0; }
    else if (rank==4) { *nData = 13; *nAPAP_G_S = 3; data->BW = 70.00; data->D_oral = 1000.0; data->D_oral_rel=0; }
    else if (rank==5) { *nData = 9;  *nAPAP_G_S = 3; data->BW = 60.00; data->D_oral = 1000.0; data->D_oral_rel=0; }
//    else if (rank==6) { *nData = 15; *nAPAP_G_S = 2; data->BW = 62.50; data->D_oral = 500.0; data->D_oral_rel=0; }
    else if (rank==6) { *nData = 15; *nAPAP_G_S = 3; data->BW = 62.50; data->D_oral = 500.0; data->D_oral_rel=0; }
    else if (rank==7) { *nData = 11; *nAPAP_G_S = 3; data->BW = 75.00; data->D_oral = 325.0; data->D_oral_rel=0; }
//    else if (rank==8) { *nData = 12; *nAPAP_G_S = 1; data->BW = 62.70; data->D_oral = 500.0; data->D_oral_rel=0; }
//    else if (rank==9) { *nData = 14; *nAPAP_G_S = 1; data->BW = 60.00; data->D_oral = 650.0; data->D_oral_rel=0; }
    else if (rank==8) { *nData = 12; *nAPAP_G_S = 3; data->BW = 62.70; data->D_oral = 500.0; data->D_oral_rel=0; }
    else if (rank==9) { *nData = 14; *nAPAP_G_S = 3; data->BW = 60.00; data->D_oral = 650.0; data->D_oral_rel=0; }
    else {    return(1);  }
    return(0);
}
int getDataSynthetic(int rank, N_Vector data2fit){
    // synthetische Daten: Werte der Lösung, die ich mit den Posterior-Parametern von Zurlinden erhalten habe.
    int kk = 1;

    switch (rank) {
        case 0:
            // Chiew 2010 (mikro mol / Liter): nData=14; nAPAP_G_S=3;
            //  80mg/kg oral. meanBW = 73.11 (range=62-84) -> meanAPAPdosis = 79mg/kg (range=77-83), meanAge = 34 (range=27-46)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 1.6411712867e+02;	Ith(data2fit,kk++) = 3.0148052130e+01;	Ith(data2fit,kk++) = 3.9924968234e+01;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 1.5916839292e+02;	Ith(data2fit,kk++) = 5.3418543016e+01;	Ith(data2fit,kk++) = 4.9408752047e+01;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 1.4129312531e+02;	Ith(data2fit,kk++) = 7.2880208366e+01;	Ith(data2fit,kk++) = 5.2886457428e+01;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 9.9947347904e+01;	Ith(data2fit,kk++) = 9.3658170825e+01;	Ith(data2fit,kk++) = 4.9206397861e+01;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 6.6832781103e+01;	Ith(data2fit,kk++) = 9.3332326396e+01;	Ith(data2fit,kk++) = 3.9648388255e+01;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 2.8495910803e+01;	Ith(data2fit,kk++) = 6.4672293906e+01;	Ith(data2fit,kk++) = 2.1055590647e+01;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 1.1949695512e+01;	Ith(data2fit,kk++) = 3.5387217843e+01;	Ith(data2fit,kk++) = 9.8480481706e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 2.0809632105e+00;	Ith(data2fit,kk++) = 7.9146720420e+00;	Ith(data2fit,kk++) = 1.8654067526e+00;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 3.6138088006e-01;	Ith(data2fit,kk++) = 1.5159248719e+00;	Ith(data2fit,kk++) = 3.3120911306e-01;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 6.2728694811e-02;	Ith(data2fit,kk++) = 2.7425748691e-01;	Ith(data2fit,kk++) = 5.7823079859e-02;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 1.0887615890e-02;	Ith(data2fit,kk++) = 4.8473130562e-02;	Ith(data2fit,kk++) = 1.0050814163e-02;
            Ith(data2fit,kk++) = 1.6000000000e+01;	Ith(data2fit,kk++) = 3.2798413886e-04;	Ith(data2fit,kk++) = 1.4774159857e-03;	Ith(data2fit,kk++) = 3.0291289423e-04;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 2.9763914415e-07;	Ith(data2fit,kk++) = 1.3445520401e-06;	Ith(data2fit,kk++) = 2.7489532542e-07;
            break;
        case 1:
            // Critchley 2005 Caucasian (mikro gramm / milli Liter): nData=15; nAPAP_G_S=3;
            // 20mg/kg oral. meanBW = 68 (range=55-97), meanAge = 29 (range=23-44), meanHeight =  175 (range=166-187)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 2.5000000000e-01;	Ith(data2fit,kk++) = 3.2205527553e+01;	Ith(data2fit,kk++) = 2.1973648188e+00;	Ith(data2fit,kk++) = 6.2171717605e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 3.9291775595e+01;	Ith(data2fit,kk++) = 7.6001694440e+00;	Ith(data2fit,kk++) = 1.1177040453e+01;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 3.7856269797e+01;	Ith(data2fit,kk++) = 1.3452121540e+01;	Ith(data2fit,kk++) = 1.3709182810e+01;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 3.3407728166e+01;	Ith(data2fit,kk++) = 1.8275996971e+01;	Ith(data2fit,kk++) = 1.4487628788e+01;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 2.3403783048e+01;	Ith(data2fit,kk++) = 2.3202088303e+01;	Ith(data2fit,kk++) = 1.3111745387e+01;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 1.5539483053e+01;	Ith(data2fit,kk++) = 2.2839908553e+01;	Ith(data2fit,kk++) = 1.0304484241e+01;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 6.5727445463e+00;	Ith(data2fit,kk++) = 1.5564392685e+01;	Ith(data2fit,kk++) = 5.2753617114e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 2.7477296692e+00;	Ith(data2fit,kk++) = 8.4601631164e+00;	Ith(data2fit,kk++) = 2.4141197864e+00;
            Ith(data2fit,kk++) = 5.0000000000e+00;	Ith(data2fit,kk++) = 1.1465324128e+00;	Ith(data2fit,kk++) = 4.1237947492e+00;	Ith(data2fit,kk++) = 1.0528952774e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 4.7820137526e-01;	Ith(data2fit,kk++) = 1.8942950359e+00;	Ith(data2fit,kk++) = 4.4894440291e-01;
            Ith(data2fit,kk++) = 7.0000000000e+00;	Ith(data2fit,kk++) = 1.9942271923e-01;	Ith(data2fit,kk++) = 8.4042452839e-01;	Ith(data2fit,kk++) = 1.8931392689e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 8.3160139307e-02;	Ith(data2fit,kk++) = 3.6497134246e-01;	Ith(data2fit,kk++) = 7.9389539417e-02;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 2.5143406813e-03;	Ith(data2fit,kk++) = 1.1783978614e-02;	Ith(data2fit,kk++) = 2.4132743752e-03;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 6.9489268382e-08;	Ith(data2fit,kk++) = 3.3152085350e-07;	Ith(data2fit,kk++) = 6.6720915346e-08;
            break;
        case 2:
            // Critchley 2005 Chinese (mikro gramm / milli Liter): nData=15; nAPAP_G_S=3;
            // 20mg/kg oral. meanBW = 57 (range=46-71), meanAge = 24 (range=21-32), meanHeight =  163 (range=154-178)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 2.5000000000e-01;	Ith(data2fit,kk++) = 1.2518409121e+02;	Ith(data2fit,kk++) = 8.3302871555e+00;	Ith(data2fit,kk++) = 2.1621356562e+01;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 1.5381110001e+02;	Ith(data2fit,kk++) = 2.8892486847e+01;	Ith(data2fit,kk++) = 3.9126069932e+01;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 1.4910578601e+02;	Ith(data2fit,kk++) = 5.1502133342e+01;	Ith(data2fit,kk++) = 4.8795030135e+01;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 1.3232115952e+02;	Ith(data2fit,kk++) = 7.0627975582e+01;	Ith(data2fit,kk++) = 5.2569428689e+01;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 9.3585161819e+01;	Ith(data2fit,kk++) = 9.1603376528e+01;	Ith(data2fit,kk++) = 4.9460283612e+01;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 6.2599237954e+01;	Ith(data2fit,kk++) = 9.2119288742e+01;	Ith(data2fit,kk++) = 4.0251136834e+01;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 2.6736433520e+01;	Ith(data2fit,kk++) = 6.5115415247e+01;	Ith(data2fit,kk++) = 2.1757426281e+01;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 1.1239670694e+01;	Ith(data2fit,kk++) = 3.6404100476e+01;	Ith(data2fit,kk++) = 1.0329224886e+01;
            Ith(data2fit,kk++) = 5.0000000000e+00;	Ith(data2fit,kk++) = 4.7069473213e+00;	Ith(data2fit,kk++) = 1.8139286087e+01;	Ith(data2fit,kk++) = 4.6153544570e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 1.9686359775e+00;	Ith(data2fit,kk++) = 8.4814260773e+00;	Ith(data2fit,kk++) = 1.9993463425e+00;
            Ith(data2fit,kk++) = 7.0000000000e+00;	Ith(data2fit,kk++) = 8.2294906216e-01;	Ith(data2fit,kk++) = 3.8181581852e+00;	Ith(data2fit,kk++) = 8.5195221382e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 3.4394649032e-01;	Ith(data2fit,kk++) = 1.6783552797e+00;	Ith(data2fit,kk++) = 3.5980961804e-01;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 1.0489412122e-02;	Ith(data2fit,kk++) = 5.5954166809e-02;	Ith(data2fit,kk++) = 1.1098597277e-02;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 2.9744036582e-07;	Ith(data2fit,kk++) = 1.6360021831e-06;	Ith(data2fit,kk++) = 3.1504597974e-07;
            break;
        case 3:
            // Jensen 2004 (mikro Mol): nData=13; nAPAP_G_S=3;
            //  1000mg oral meanBW = ...
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 2.7877670550e+01;	Ith(data2fit,kk++) = 5.4068281241e+00;	Ith(data2fit,kk++) = 8.0085496876e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 2.3672145184e+01;	Ith(data2fit,kk++) = 1.2971149691e+01;	Ith(data2fit,kk++) = 1.0333446847e+01;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 1.6566071930e+01;	Ith(data2fit,kk++) = 1.6419653645e+01;	Ith(data2fit,kk++) = 9.3086777834e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 1.0990353865e+01;	Ith(data2fit,kk++) = 1.6117085722e+01;	Ith(data2fit,kk++) = 7.2853096365e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 4.6433728223e+00;	Ith(data2fit,kk++) = 1.0928631783e+01;	Ith(data2fit,kk++) = 3.7054175625e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 1.9397947726e+00;	Ith(data2fit,kk++) = 5.9169913035e+00;	Ith(data2fit,kk++) = 1.6881192991e+00;
            Ith(data2fit,kk++) = 5.0000000000e+00;	Ith(data2fit,kk++) = 8.0899912622e-01;	Ith(data2fit,kk++) = 2.8750400739e+00;	Ith(data2fit,kk++) = 7.3407855389e-01;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 3.3727862066e-01;	Ith(data2fit,kk++) = 1.3172521235e+00;	Ith(data2fit,kk++) = 3.1239468404e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 5.8608847685e-02;	Ith(data2fit,kk++) = 2.5279799922e-01;	Ith(data2fit,kk++) = 5.5120831579e-02;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 1.0183675729e-02;	Ith(data2fit,kk++) = 4.5859837717e-02;	Ith(data2fit,kk++) = 9.6141528729e-03;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 1.7694585739e-03;	Ith(data2fit,kk++) = 8.1242104752e-03;	Ith(data2fit,kk++) = 1.6720869841e-03;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 4.8690988939e-08;	Ith(data2fit,kk++) = 2.2722249970e-07;	Ith(data2fit,kk++) = 4.6025972826e-08;
            break;
        case 4:
            // Kim 2010 (mikro gramm/milli Liter): nData=13; nAPAP_G_S=3;
            // 1000mg oral meanBW = 70 ± 15 (range=51-105), meanAge = 44 ± 12 (range=27-68)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 2.5000000000e-01;	Ith(data2fit,kk++) = 2.2867987087e+01;	Ith(data2fit,kk++) = 1.5640954586e+00;	Ith(data2fit,kk++) = 4.4605989191e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 2.7877670550e+01;	Ith(data2fit,kk++) = 5.4068281241e+00;	Ith(data2fit,kk++) = 8.0085496876e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 2.3672145183e+01;	Ith(data2fit,kk++) = 1.2971149691e+01;	Ith(data2fit,kk++) = 1.0333446847e+01;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 1.6566071930e+01;	Ith(data2fit,kk++) = 1.6419653645e+01;	Ith(data2fit,kk++) = 9.3086777834e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 1.0990353865e+01;	Ith(data2fit,kk++) = 1.6117085722e+01;	Ith(data2fit,kk++) = 7.2853096364e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 4.6433728223e+00;	Ith(data2fit,kk++) = 1.0928631783e+01;	Ith(data2fit,kk++) = 3.7054175625e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 1.9397947726e+00;	Ith(data2fit,kk++) = 5.9169913035e+00;	Ith(data2fit,kk++) = 1.6881192991e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 3.3727862066e-01;	Ith(data2fit,kk++) = 1.3172521235e+00;	Ith(data2fit,kk++) = 3.1239468404e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 5.8608847683e-02;	Ith(data2fit,kk++) = 2.5279799922e-01;	Ith(data2fit,kk++) = 5.5120831579e-02;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 1.0183675728e-02;	Ith(data2fit,kk++) = 4.5859837716e-02;	Ith(data2fit,kk++) = 9.6141528727e-03;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 1.7694585738e-03;	Ith(data2fit,kk++) = 8.1242104750e-03;	Ith(data2fit,kk++) = 1.6720869840e-03;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 4.8690988936e-08;	Ith(data2fit,kk++) = 2.2722249968e-07;	Ith(data2fit,kk++) = 4.6025972822e-08;
            break;
        case 5:
            // Shinoda 2007 (mikro gramm / milli Liter): nData=9; nAPAP_G_S=3;
            // 1000mg oral. meanBW = 60
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 2.5000000000e-01;	Ith(data2fit,kk++) = 2.5689892183e+01;	Ith(data2fit,kk++) = 1.7637210397e+00;	Ith(data2fit,kk++) = 5.0319405203e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 3.1332706546e+01;	Ith(data2fit,kk++) = 6.1210588615e+00;	Ith(data2fit,kk++) = 9.0920588983e+00;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 3.0181907156e+01;	Ith(data2fit,kk++) = 1.0865868647e+01;	Ith(data2fit,kk++) = 1.1197161049e+01;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 2.6631886971e+01;	Ith(data2fit,kk++) = 1.4802134849e+01;	Ith(data2fit,kk++) = 1.1875810049e+01;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 1.2389890221e+01;	Ith(data2fit,kk++) = 1.8692102056e+01;	Ith(data2fit,kk++) = 8.5518732162e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 5.2452844156e+00;	Ith(data2fit,kk++) = 1.2879069692e+01;	Ith(data2fit,kk++) = 4.4224384757e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 2.1955521568e+00;	Ith(data2fit,kk++) = 7.0795288250e+00;	Ith(data2fit,kk++) = 2.0399958158e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 3.8321215521e-01;	Ith(data2fit,kk++) = 1.6179196516e+00;	Ith(data2fit,kk++) = 3.8347274477e-01;
            break;
        case 6:
            // Tan 2012 (mikro gramm / milli Liter): nData=15; nAPAP_G_S=2;
            // 500mg oral meanBW = 62.5 [5.6], meanAge = 23.7 [1.8]
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 1.6667000000e-01;	Ith(data2fit,kk++) = 7.0779190304e+00;	Ith(data2fit,kk++) = 2.6250821875e-01;	Ith(data2fit,kk++) = 1.1232748612e+00;
            Ith(data2fit,kk++) = 3.3333000000e-01;	Ith(data2fit,kk++) = 9.9177454565e+00;	Ith(data2fit,kk++) = 1.0578202355e+00;	Ith(data2fit,kk++) = 2.3254171746e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 1.0778959602e+01;	Ith(data2fit,kk++) = 2.1168989775e+00;	Ith(data2fit,kk++) = 3.1843620205e+00;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 1.0370662816e+01;	Ith(data2fit,kk++) = 3.7518761033e+00;	Ith(data2fit,kk++) = 3.9083286561e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 9.1410531460e+00;	Ith(data2fit,kk++) = 5.1006618291e+00;	Ith(data2fit,kk++) = 4.1297529825e+00;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 6.3919687238e+00;	Ith(data2fit,kk++) = 6.4795495355e+00;	Ith(data2fit,kk++) = 3.7343761280e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 4.2390388296e+00;	Ith(data2fit,kk++) = 6.3833768090e+00;	Ith(data2fit,kk++) = 2.9326739217e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 1.7912933853e+00;	Ith(data2fit,kk++) = 4.3647314494e+00;	Ith(data2fit,kk++) = 1.5008752872e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 7.4897054825e-01;	Ith(data2fit,kk++) = 2.3849709563e+00;	Ith(data2fit,kk++) = 6.8737820052e-01;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 1.3054840102e-01;	Ith(data2fit,kk++) = 5.4033632716e-01;	Ith(data2fit,kk++) = 1.2818328156e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 2.2747692566e-02;	Ith(data2fit,kk++) = 1.0516475184e-01;	Ith(data2fit,kk++) = 2.2730362018e-02;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 3.9635996729e-03;	Ith(data2fit,kk++) = 1.9277675107e-02;	Ith(data2fit,kk++) = 3.9790862741e-03;
            Ith(data2fit,kk++) = 1.2000000000e+01;	Ith(data2fit,kk++) = 6.9062160687e-04;	Ith(data2fit,kk++) = 3.4412044957e-03;	Ith(data2fit,kk++) = 6.9418518158e-04;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 1.9325896714e-08;	Ith(data2fit,kk++) = 9.8515357833e-08;	Ith(data2fit,kk++) = 1.9434467905e-08;

//            Ith(data2fit,kk++) = 0.0000e+00;    Ith(data2fit,kk++) =   0.000000e+00;    Ith(data2fit,kk++) =   0.000000e+00;    // Ith(data2fit,kk++) =   0.000000e+00;
//            Ith(data2fit,kk++) = 1.6667e-01;    Ith(data2fit,kk++) =   7.077829e+00;    Ith(data2fit,kk++) =   2.624968e-01;    // Ith(data2fit,kk++) =   1.123248e+00;
//            Ith(data2fit,kk++) = 3.3333e-01;    Ith(data2fit,kk++) =   9.917778e+00;    Ith(data2fit,kk++) =   1.057840e+00;    // Ith(data2fit,kk++) =   2.325438e+00;
//            Ith(data2fit,kk++) = 5.0000e-01;    Ith(data2fit,kk++) =   1.077896e+01;    Ith(data2fit,kk++) =   2.116899e+00;    // Ith(data2fit,kk++) =   3.184362e+00;
//            Ith(data2fit,kk++) = 7.5000e-01;    Ith(data2fit,kk++) =   1.037066e+01;    Ith(data2fit,kk++) =   3.751876e+00;    // Ith(data2fit,kk++) =   3.908329e+00;
//            Ith(data2fit,kk++) = 1.0000e+00;    Ith(data2fit,kk++) =   9.141053e+00;    Ith(data2fit,kk++) =   5.100662e+00;    // Ith(data2fit,kk++) =   4.129753e+00;
//            Ith(data2fit,kk++) = 1.5000e+00;    Ith(data2fit,kk++) =   6.391969e+00;    Ith(data2fit,kk++) =   6.479550e+00;    // Ith(data2fit,kk++) =   3.734376e+00;
//            Ith(data2fit,kk++) = 2.0000e+00;    Ith(data2fit,kk++) =   4.239039e+00;    Ith(data2fit,kk++) =   6.383377e+00;    // Ith(data2fit,kk++) =   2.932674e+00;
//            Ith(data2fit,kk++) = 3.0000e+00;    Ith(data2fit,kk++) =   1.791293e+00;    Ith(data2fit,kk++) =   4.364731e+00;    // Ith(data2fit,kk++) =   1.500875e+00;
//            Ith(data2fit,kk++) = 4.0000e+00;    Ith(data2fit,kk++) =   7.489705e-01;    Ith(data2fit,kk++) =   2.384971e+00;    // Ith(data2fit,kk++) =   6.873782e-01;
//            Ith(data2fit,kk++) = 6.0000e+00;    Ith(data2fit,kk++) =   1.305484e-01;    Ith(data2fit,kk++) =   5.403363e-01;    // Ith(data2fit,kk++) =   1.281833e-01;
//            Ith(data2fit,kk++) = 8.0000e+00;    Ith(data2fit,kk++) =   2.274769e-02;    Ith(data2fit,kk++) =   1.051648e-01;    // Ith(data2fit,kk++) =   2.273036e-02;
//            Ith(data2fit,kk++) = 1.0000e+01;    Ith(data2fit,kk++) =   3.963600e-03;    Ith(data2fit,kk++) =   1.927768e-02;    // Ith(data2fit,kk++) =   3.979086e-03;
//            Ith(data2fit,kk++) = 1.2000e+01;    Ith(data2fit,kk++) =   6.906216e-04;    Ith(data2fit,kk++) =   3.441204e-03;    // Ith(data2fit,kk++) =   6.941852e-04;
//            Ith(data2fit,kk++) = 2.4000e+01;    Ith(data2fit,kk++) =   1.932590e-08;    Ith(data2fit,kk++) =   9.851536e-08;    // Ith(data2fit,kk++) =   1.943447e-08;
            break;
        case 7:
            // Volak 2013 (mikro gramm / milli Liter): nData=11; nAPAP_G_S=3;
            // 325mg oral meanBW =  (range:61-93), meanAge = (range:24-52)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 5.2378954196e+00;	Ith(data2fit,kk++) = 1.0210444522e+00;	Ith(data2fit,kk++) = 1.5327641716e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 4.4356830082e+00;	Ith(data2fit,kk++) = 2.4363342607e+00;	Ith(data2fit,kk++) = 1.9579570075e+00;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 3.0973366834e+00;	Ith(data2fit,kk++) = 3.0639410013e+00;	Ith(data2fit,kk++) = 1.7460928811e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 2.0512955108e+00;	Ith(data2fit,kk++) = 2.9881663114e+00;	Ith(data2fit,kk++) = 1.3543164133e+00;
            Ith(data2fit,kk++) = 2.5000000000e+00;	Ith(data2fit,kk++) = 1.3355803351e+00;	Ith(data2fit,kk++) = 2.5480117799e+00;	Ith(data2fit,kk++) = 9.7864806564e-01;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 8.6458795685e-01;	Ith(data2fit,kk++) = 2.0036188972e+00;	Ith(data2fit,kk++) = 6.7914707953e-01;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 3.6062666038e-01;	Ith(data2fit,kk++) = 1.0751273954e+00;	Ith(data2fit,kk++) = 3.0642528082e-01;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 6.2568432370e-02;	Ith(data2fit,kk++) = 2.3621697765e-01;	Ith(data2fit,kk++) = 5.6106389311e-02;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 1.0852884258e-02;	Ith(data2fit,kk++) = 4.4928613116e-02;	Ith(data2fit,kk++) = 9.8523578494e-03;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 1.8824686590e-03;	Ith(data2fit,kk++) = 8.1002461916e-03;	Ith(data2fit,kk++) = 1.7138235460e-03;
            break;
        case 8:
            // Yin 2001 (mikro gramm / milli Liter): nData=12; nAPAP_G_S=1;
            // 500mg oral meanBW = 62.7 ± 5.6, meanAge = 24.1 ± 7.1, meanHeight = 170.6 ± 4.4
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 2.5000000000e-01;	Ith(data2fit,kk++) = 8.8288232059e+00;	Ith(data2fit,kk++) = 6.0890028594e-01;	Ith(data2fit,kk++) = 1.7613269593e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 1.0752962551e+01;	Ith(data2fit,kk++) = 2.1114636872e+00;	Ith(data2fit,kk++) = 3.1759227753e+00;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 1.0345566185e+01;	Ith(data2fit,kk++) = 3.7419404576e+00;	Ith(data2fit,kk++) = 3.8974787508e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 9.1188534878e+00;	Ith(data2fit,kk++) = 5.0867398830e+00;	Ith(data2fit,kk++) = 4.1177913294e+00;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 6.3763282824e+00;	Ith(data2fit,kk++) = 6.4607993364e+00;	Ith(data2fit,kk++) = 3.7227222020e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 4.2285849490e+00;	Ith(data2fit,kk++) = 6.3638480433e+00;	Ith(data2fit,kk++) = 2.9229296527e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 1.7868056375e+00;	Ith(data2fit,kk++) = 4.3499483693e+00;	Ith(data2fit,kk++) = 1.4953832563e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 7.4706479549e-01;	Ith(data2fit,kk++) = 2.3761497989e+00;	Ith(data2fit,kk++) = 6.8468851283e-01;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 1.3020606238e-01;	Ith(data2fit,kk++) = 5.3804483972e-01;	Ith(data2fit,kk++) = 1.2764045520e-01;
            Ith(data2fit,kk++) = 8.0000000000e+00;	Ith(data2fit,kk++) = 2.2686277259e-02;	Ith(data2fit,kk++) = 1.0467470493e-01;	Ith(data2fit,kk++) = 2.2630078572e-02;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 3.9525914788e-03;	Ith(data2fit,kk++) = 1.9181875383e-02;	Ith(data2fit,kk++) = 3.9610772974e-03;

//            Ith(data2fit,kk++) = 0.0000e+00;    Ith(data2fit,kk++) =   0.000000e+00;    // Ith(data2fit,kk++) =   0.000000e+00;    Ith(data2fit,kk++) =   0.000000e+00;
//            Ith(data2fit,kk++) = 2.5000e-01;    Ith(data2fit,kk++) =   8.828823e+00;    // Ith(data2fit,kk++) =   6.089003e-01;    Ith(data2fit,kk++) =   1.761327e+00;
//            Ith(data2fit,kk++) = 5.0000e-01;    Ith(data2fit,kk++) =   1.075296e+01;    // Ith(data2fit,kk++) =   2.111464e+00;    Ith(data2fit,kk++) =   3.175923e+00;
//            Ith(data2fit,kk++) = 7.5000e-01;    Ith(data2fit,kk++) =   1.034557e+01;    // Ith(data2fit,kk++) =   3.741940e+00;    Ith(data2fit,kk++) =   3.897479e+00;
//            Ith(data2fit,kk++) = 1.0000e+00;    Ith(data2fit,kk++) =   9.118853e+00;    // Ith(data2fit,kk++) =   5.086740e+00;    Ith(data2fit,kk++) =   4.117791e+00;
//            Ith(data2fit,kk++) = 1.5000e+00;    Ith(data2fit,kk++) =   6.376328e+00;    // Ith(data2fit,kk++) =   6.460799e+00;    Ith(data2fit,kk++) =   3.722722e+00;
//            Ith(data2fit,kk++) = 2.0000e+00;    Ith(data2fit,kk++) =   4.228585e+00;    // Ith(data2fit,kk++) =   6.363848e+00;    Ith(data2fit,kk++) =   2.922930e+00;
//            Ith(data2fit,kk++) = 3.0000e+00;    Ith(data2fit,kk++) =   1.786806e+00;    // Ith(data2fit,kk++) =   4.349948e+00;    Ith(data2fit,kk++) =   1.495383e+00;
//            Ith(data2fit,kk++) = 4.0000e+00;    Ith(data2fit,kk++) =   7.470648e-01;    // Ith(data2fit,kk++) =   2.376150e+00;    Ith(data2fit,kk++) =   6.846885e-01;
//            Ith(data2fit,kk++) = 6.0000e+00;    Ith(data2fit,kk++) =   1.302061e-01;    // Ith(data2fit,kk++) =   5.380448e-01;    Ith(data2fit,kk++) =   1.276405e-01;
//            Ith(data2fit,kk++) = 8.0000e+00;    Ith(data2fit,kk++) =   2.268628e-02;    // Ith(data2fit,kk++) =   1.046747e-01;    Ith(data2fit,kk++) =   2.263008e-02;
//            Ith(data2fit,kk++) = 1.0000e+01;    Ith(data2fit,kk++) =   3.952591e-03;    // Ith(data2fit,kk++) =   1.918188e-02;    Ith(data2fit,kk++) =   3.961077e-03;
            break;
        case 9:
            // Zhu 2007 (mikro gramm / milli Liter): nData=14; nAPAP_G_S=1;
            // 650mg oral meanBW = 60 (Chinese)
            Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;	Ith(data2fit,kk++) = 0.0000000000e+00;
            Ith(data2fit,kk++) = 1.6667000000e-01;	Ith(data2fit,kk++) = 1.0642700591e+01;	Ith(data2fit,kk++) = 3.9454644205e-01;	Ith(data2fit,kk++) = 1.6829549690e+00;
            Ith(data2fit,kk++) = 3.3333000000e-01;	Ith(data2fit,kk++) = 1.4917831778e+01;	Ith(data2fit,kk++) = 1.5907262853e+00;	Ith(data2fit,kk++) = 3.4867448881e+00;
            Ith(data2fit,kk++) = 5.0000000000e-01;	Ith(data2fit,kk++) = 1.6217716889e+01;	Ith(data2fit,kk++) = 3.1854833284e+00;	Ith(data2fit,kk++) = 4.7806065471e+00;
            Ith(data2fit,kk++) = 7.5000000000e-01;	Ith(data2fit,kk++) = 1.5609358358e+01;	Ith(data2fit,kk++) = 5.6524031357e+00;	Ith(data2fit,kk++) = 5.8795760830e+00;
            Ith(data2fit,kk++) = 1.0000000000e+00;	Ith(data2fit,kk++) = 1.3763469717e+01;	Ith(data2fit,kk++) = 7.6943510172e+00;	Ith(data2fit,kk++) = 6.2256681380e+00;
            Ith(data2fit,kk++) = 1.5000000000e+00;	Ith(data2fit,kk++) = 9.6303350658e+00;	Ith(data2fit,kk++) = 9.8010151652e+00;	Ith(data2fit,kk++) = 5.6521297685e+00;
            Ith(data2fit,kk++) = 2.0000000000e+00;	Ith(data2fit,kk++) = 6.3900947267e+00;	Ith(data2fit,kk++) = 9.6817557888e+00;	Ith(data2fit,kk++) = 4.4546819824e+00;
            Ith(data2fit,kk++) = 3.0000000000e+00;	Ith(data2fit,kk++) = 2.7025318357e+00;	Ith(data2fit,kk++) = 6.6536782968e+00;	Ith(data2fit,kk++) = 2.2931756323e+00;
            Ith(data2fit,kk++) = 4.0000000000e+00;	Ith(data2fit,kk++) = 1.1307098134e+00;	Ith(data2fit,kk++) = 3.6520942965e+00;	Ith(data2fit,kk++) = 1.0547182874e+00;
            Ith(data2fit,kk++) = 6.0000000000e+00;	Ith(data2fit,kk++) = 1.9730107388e-01;	Ith(data2fit,kk++) = 8.3357467634e-01;	Ith(data2fit,kk++) = 1.9769825915e-01;
            Ith(data2fit,kk++) = 1.0000000000e+01;	Ith(data2fit,kk++) = 6.0021836312e-03;	Ith(data2fit,kk++) = 3.0028914173e-02;	Ith(data2fit,kk++) = 6.1628665488e-03;
            Ith(data2fit,kk++) = 1.5000000000e+01;	Ith(data2fit,kk++) = 7.6252014481e-05;	Ith(data2fit,kk++) = 3.9826002708e-04;	Ith(data2fit,kk++) = 7.8443861097e-05;
            Ith(data2fit,kk++) = 2.4000000000e+01;	Ith(data2fit,kk++) = 2.9467654839e-08;	Ith(data2fit,kk++) = 1.5528456284e-07;	Ith(data2fit,kk++) = 3.0317106242e-08;

//            Ith(data2fit,kk++) = 0.0000e+00;    Ith(data2fit,kk++) =   0.000000e+00;    // // Ith(data2fit,kk++) =   0.000000e+00;    Ith(data2fit,kk++) =   0.000000e+00;
//            Ith(data2fit,kk++) = 1.6667e-01;    Ith(data2fit,kk++) =   1.064256e+01;    // Ith(data2fit,kk++) =   3.945293e-01;    Ith(data2fit,kk++) =   1.682914e+00;
//            Ith(data2fit,kk++) = 3.3333e-01;    Ith(data2fit,kk++) =   1.491788e+01;    // Ith(data2fit,kk++) =   1.590755e+00;    Ith(data2fit,kk++) =   3.486776e+00;
//            Ith(data2fit,kk++) = 5.0000e-01;    Ith(data2fit,kk++) =   1.621772e+01;    // Ith(data2fit,kk++) =   3.185483e+00;    Ith(data2fit,kk++) =   4.780607e+00;
//            Ith(data2fit,kk++) = 7.5000e-01;    Ith(data2fit,kk++) =   1.560936e+01;    // Ith(data2fit,kk++) =   5.652403e+00;    Ith(data2fit,kk++) =   5.879576e+00;
//            Ith(data2fit,kk++) = 1.0000e+00;    Ith(data2fit,kk++) =   1.376347e+01;    // Ith(data2fit,kk++) =   7.694351e+00;    Ith(data2fit,kk++) =   6.225668e+00;
//            Ith(data2fit,kk++) = 1.5000e+00;    Ith(data2fit,kk++) =   9.630335e+00;    // Ith(data2fit,kk++) =   9.801015e+00;    Ith(data2fit,kk++) =   5.652130e+00;
//            Ith(data2fit,kk++) = 2.0000e+00;    Ith(data2fit,kk++) =   6.390095e+00;    // Ith(data2fit,kk++) =   9.681756e+00;    Ith(data2fit,kk++) =   4.454682e+00;
//            Ith(data2fit,kk++) = 3.0000e+00;    Ith(data2fit,kk++) =   2.702532e+00;    // Ith(data2fit,kk++) =   6.653678e+00;    Ith(data2fit,kk++) =   2.293176e+00;
//            Ith(data2fit,kk++) = 4.0000e+00;    Ith(data2fit,kk++) =   1.130710e+00;    // Ith(data2fit,kk++) =   3.652094e+00;    Ith(data2fit,kk++) =   1.054718e+00;
//            Ith(data2fit,kk++) = 6.0000e+00;    Ith(data2fit,kk++) =   1.973011e-01;    // Ith(data2fit,kk++) =   8.335747e-01;    Ith(data2fit,kk++) =   1.976983e-01;
//            Ith(data2fit,kk++) = 1.0000e+01;    Ith(data2fit,kk++) =   6.002184e-03;    // Ith(data2fit,kk++) =   3.002891e-02;    Ith(data2fit,kk++) =   6.162867e-03;
//            Ith(data2fit,kk++) = 1.5000e+01;    Ith(data2fit,kk++) =   7.625201e-05;    // Ith(data2fit,kk++) =   3.982600e-04;    Ith(data2fit,kk++) =   7.844386e-05;
//            Ith(data2fit,kk++) = 2.4000e+01;    Ith(data2fit,kk++) =   2.946765e-08;    // Ith(data2fit,kk++) =   1.552846e-07;    Ith(data2fit,kk++) =   3.031711e-08;
            break;
        default: return(1); break;
    }
    return(0);
}
int getData(int rank, N_Vector data2fit){
    // Daten mit R-Packet "digitize" digitalisiert (https://cran.r-project.org/web/packages/digitize/digitize.pdf https://lukemiller.org/index.php/2011/06/digitizing-data-from-old-plots-using-digitize/)
    int kk = 1;

    switch (rank) {
        case 0:
            // Chiew 2010 (mikro mol / Liter): nData=14; nAPAP_G_S=3;
            //  80mg/kg oral. meanBW = 73.11 (range=62-84) -> meanAPAPdosis = 79mg/kg (range=77-83), meanAge = 34 (range=27-46)
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 460.227273; Ith(data2fit,kk++) = 33.68421 ; Ith(data2fit,kk++) = 23.85686;
            Ith(data2fit,kk++) = 0.75 ; Ith(data2fit,kk++) = 565.599174; Ith(data2fit,kk++) = 74.10526 ; Ith(data2fit,kk++) = 42.94235;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 464.876033; Ith(data2fit,kk++) = 112.00000; Ith(data2fit,kk++) = 51.68986;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 387.396694; Ith(data2fit,kk++) = 183.57895; Ith(data2fit,kk++) = 63.61829;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 387.396694; Ith(data2fit,kk++) = 245.05263; Ith(data2fit,kk++) = 73.95626;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 348.657025; Ith(data2fit,kk++) = 314.94737; Ith(data2fit,kk++) = 85.08946;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 280.475207; Ith(data2fit,kk++) = 330.10526; Ith(data2fit,kk++) = 93.04175;
            Ith(data2fit,kk++) = 6.0 ;  Ith(data2fit,kk++) = 173.553719; Ith(data2fit,kk++) = 282.94737; Ith(data2fit,kk++) = 85.88469;
            Ith(data2fit,kk++) = 8.0 ;  Ith(data2fit,kk++) = 100.723140; Ith(data2fit,kk++) = 199.57895; Ith(data2fit,kk++) = 69.98012;
            Ith(data2fit,kk++) = 10.0     ; Ith(data2fit,kk++) = 58.884298; Ith(data2fit,kk++) = 138.94737 ; Ith(data2fit,kk++) = 52.48509;
            Ith(data2fit,kk++) = 12.0;    Ith(data2fit,kk++) = 34.090909; Ith(data2fit,kk++) = 91.78947 ; Ith(data2fit,kk++) = 37.37575;
            Ith(data2fit,kk++) = 16.0;    Ith(data2fit,kk++) = 18.595041; Ith(data2fit,kk++) = 45.47368 ; Ith(data2fit,kk++) = 20.67594;
            Ith(data2fit,kk++) = 24.0;    Ith(data2fit,kk++) = 7.747934; Ith(data2fit,kk++) = 12.63158 ; Ith(data2fit,kk++) = 5.56660;
            for (int i = 1; i < kk; i++) {
                if (i%4 == 1){}
                else {
                    Ith(data2fit,i) = Ith(data2fit,i) * 0.15116; // Einheiten auf mikro gramm/milli liter umrechnen
                }
            }
            break;
        case 1:
            // Critchley 2005 Caucasian (mikro gramm / milli Liter): nData=15; nAPAP_G_S=3;
            // 20mg/kg oral. meanBW = 68 (range=55-97), meanAge = 29 (range=23-44), meanHeight =  175 (range=166-187)
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.25 ; Ith(data2fit,kk++) = 8.88372093; Ith(data2fit,kk++) = 0.33582090 ; Ith(data2fit,kk++) = 0.7775701;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 14.74418605; Ith(data2fit,kk++) = 1.76305970 ; Ith(data2fit,kk++) = 1.9887850;
            Ith(data2fit,kk++) = 0.75 ; Ith(data2fit,kk++) = 16.83720930; Ith(data2fit,kk++) = 3.63805970 ; Ith(data2fit,kk++) = 2.9457944;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 16.74418605; Ith(data2fit,kk++) = 4.72947761 ; Ith(data2fit,kk++) = 3.4093458;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 14.88372093; Ith(data2fit,kk++) = 7.77985075 ; Ith(data2fit,kk++) = 4.0672897;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 13.72093023; Ith(data2fit,kk++) = 9.37500000 ; Ith(data2fit,kk++) = 4.2317757;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 10.79069767; Ith(data2fit,kk++) = 10.80223881; Ith(data2fit,kk++) = 4.2616822;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 8.09302326; Ith(data2fit,kk++) = 10.77425373; Ith(data2fit,kk++) = 3.7233645;
            Ith(data2fit,kk++) = 5.0     ; Ith(data2fit,kk++) = 5.95348837; Ith(data2fit,kk++) = 9.48694030; Ith(data2fit,kk++) = 3.1850467;
            Ith(data2fit,kk++) = 6.0 ;  Ith(data2fit,kk++) = 4.60465116; Ith(data2fit,kk++) = 8.11567164; Ith(data2fit,kk++) = 2.6616822;
            Ith(data2fit,kk++) = 7.0 ;  Ith(data2fit,kk++) = 3.34883721; Ith(data2fit,kk++) = 6.80037313; Ith(data2fit,kk++) = 2.1233645;
            Ith(data2fit,kk++) = 8.0     ; Ith(data2fit,kk++) = 2.51162791; Ith(data2fit,kk++) = 5.23320896 ; Ith(data2fit,kk++) = 1.6299065;
            Ith(data2fit,kk++) = 12.0;    Ith(data2fit,kk++) = 1.0687732; Ith(data2fit,kk++) = 2.04291045 ; Ith(data2fit,kk++) = 0.7327103;
            Ith(data2fit,kk++) = 24.0;    Ith(data2fit,kk++) = 0.2323420; Ith(data2fit,kk++) = 0.25186567 ; Ith(data2fit,kk++) = 0.1046729;
            break;
        case 2:
            // Critchley 2005 Chinese (mikro gramm / milli Liter): nData=15; nAPAP_G_S=3;
            // 20mg/kg oral. meanBW = 57 (range=46-71), meanAge = 24 (range=21-32), meanHeight =  163 (range=154-178)
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.25 ; Ith(data2fit,kk++) = 10.09302326; Ith(data2fit,kk++) = 0.50373134 ; Ith(data2fit,kk++) = 1.2261682;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 21.95348837; Ith(data2fit,kk++) = 2.12686567 ; Ith(data2fit,kk++) = 3.0654206;
            Ith(data2fit,kk++) = 0.75 ; Ith(data2fit,kk++) = 21.72093023; Ith(data2fit,kk++) = 4.11380597 ; Ith(data2fit,kk++) = 4.0224299;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 19.48837209; Ith(data2fit,kk++) = 5.45708955 ; Ith(data2fit,kk++) = 4.4710280;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 16.32558140; Ith(data2fit,kk++) = 7.36007463 ; Ith(data2fit,kk++) = 4.8448598;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 13.76744186; Ith(data2fit,kk++) = 8.39552239 ; Ith(data2fit,kk++) = 4.9345794;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 10.79069767; Ith(data2fit,kk++) = 9.20708955; Ith(data2fit,kk++) = 4.5906542;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 8.09302326; Ith(data2fit,kk++) = 8.89925373; Ith(data2fit,kk++) = 4.0523364;
            Ith(data2fit,kk++) = 5.0     ; Ith(data2fit,kk++) = 6.13953488; Ith(data2fit,kk++) = 8.03171642; Ith(data2fit,kk++) = 3.6186916;
            Ith(data2fit,kk++) = 6.0 ;  Ith(data2fit,kk++) = 4.37209302; Ith(data2fit,kk++) = 6.54850746; Ith(data2fit,kk++) = 2.9906542;
            Ith(data2fit,kk++) = 7.0 ;  Ith(data2fit,kk++) = 3.34883721; Ith(data2fit,kk++) = 5.20522388; Ith(data2fit,kk++) = 2.5570093;
            Ith(data2fit,kk++) = 8.0     ; Ith(data2fit,kk++) = 2.32558140; Ith(data2fit,kk++) = 4.19776119 ; Ith(data2fit,kk++) = 2.0635514;
            Ith(data2fit,kk++) = 12.0;    Ith(data2fit,kk++) = 0.8828996; Ith(data2fit,kk++) = 1.70708955 ; Ith(data2fit,kk++) = 0.8672897;
            Ith(data2fit,kk++) = 24.0;    Ith(data2fit,kk++) = 0.2323420; Ith(data2fit,kk++) = 0.22388060 ; Ith(data2fit,kk++) = 0.1046729;
            break;
        case 3:
            // Jensen 2004 (mikro Mol): nData=13; nAPAP_G_S=3;
            //  1000mg oral meanBW = ...
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0; Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 70.8333333; Ith(data2fit,kk++) = 17.2916667 ; Ith(data2fit,kk++) = 13.3333333;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 85.4166667; Ith(data2fit,kk++) = 35.2083333 ; Ith(data2fit,kk++) = 28.5416667;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 71.4583333; Ith(data2fit,kk++) = 55.4166667 ; Ith(data2fit,kk++) = 33.7500000;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 57.2916667; Ith(data2fit,kk++) = 65.6250000 ; Ith(data2fit,kk++) = 31.0416667;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 40.4166667; Ith(data2fit,kk++) = 70.4166667; Ith(data2fit,kk++) = 28.3333333;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 28.9583333; Ith(data2fit,kk++) = 62.2916667; Ith(data2fit,kk++) = 22.0833333;
            Ith(data2fit,kk++) = 5.0     ; Ith(data2fit,kk++) = 23.3333333; Ith(data2fit,kk++) = 53.7500000; Ith(data2fit,kk++) = 18.9583333;
            Ith(data2fit,kk++) = 6.0   ; Ith(data2fit,kk++) = 17.2916667; Ith(data2fit,kk++) = 42.7083333; Ith(data2fit,kk++) = 15.0000000;
            Ith(data2fit,kk++) = 8.0     ; Ith(data2fit,kk++) = 10.0000000; Ith(data2fit,kk++) = 27.2916667 ; Ith(data2fit,kk++) = 9.3750000;
            Ith(data2fit,kk++) = 10.0;    Ith(data2fit,kk++) = 6.6666667; Ith(data2fit,kk++) = 17.2916667 ; Ith(data2fit,kk++) = 5.8333333;
            Ith(data2fit,kk++) = 12.0;    Ith(data2fit,kk++) = 4.3750000; Ith(data2fit,kk++) = 11.4583333 ; Ith(data2fit,kk++) = 3.9583333;
            Ith(data2fit,kk++) = 24.0;    Ith(data2fit,kk++) = 0.8333333; Ith(data2fit,kk++) = 1.2500000 ; Ith(data2fit,kk++) = 0.8333333;
            for (int i = 1; i < kk; i++) {
                if ( i%4 == 1){}
                else {
                    Ith(data2fit,i) = Ith(data2fit,i) * 0.15116; // Einheiten auf mikro gramm/milli liter umrechnen
                }
            }
            break;
        case 4:
            // Kim 2010 (mikro gramm/milli Liter): nData=13; nAPAP_G_S=3;
            // 1000mg oral meanBW = 70 ± 15 (range=51-105), meanAge = 44 ± 12 (range=27-68)
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0;       Ith(data2fit,kk++) = 0.0;         Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.25  ; Ith(data2fit,kk++) = 4.1748014; Ith(data2fit,kk++) = 0.3051643  ; Ith(data2fit,kk++) = 0.2112676;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 3.1201194; Ith(data2fit,kk++) = 0.96244131 ; Ith(data2fit,kk++) = 0.63380282;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 5.0317305; Ith(data2fit,kk++) = 3.49765258 ; Ith(data2fit,kk++) = 2.04225352;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 8.6242412; Ith(data2fit,kk++) = 6.36150235 ; Ith(data2fit,kk++) = 3.49765258;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 8.8549529; Ith(data2fit,kk++) = 9.31924883 ; Ith(data2fit,kk++) = 4.48356808;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 8.4594471; Ith(data2fit,kk++) = 14.38967136; Ith(data2fit,kk++) = 5.18779343;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 6.9433417; Ith(data2fit,kk++) = 16.36150235; Ith(data2fit,kk++) = 5.14084507;
            Ith(data2fit,kk++) = 6.0 ;   Ith(data2fit,kk++) = 3.6804192; Ith(data2fit,kk++) = 14.67136150; Ith(data2fit,kk++) = 3.73239437;
            Ith(data2fit,kk++) = 8.0     ; Ith(data2fit,kk++) = 1.8676845; Ith(data2fit,kk++) = 9.60093897 ; Ith(data2fit,kk++) = 2.27699531;
            Ith(data2fit,kk++) = 10.0;     Ith(data2fit,kk++) = 1.1096318; Ith(data2fit,kk++) = 5.89201878 ; Ith(data2fit,kk++) = 1.43192488;
            Ith(data2fit,kk++) = 12.0;     Ith(data2fit,kk++) = 0.6811672; Ith(data2fit,kk++) = 3.54460094 ; Ith(data2fit,kk++) = 0.86854460;
            Ith(data2fit,kk++) = 24.0;     Ith(data2fit,kk++) = 0.1867850; Ith(data2fit,kk++) = 0.44600939 ; Ith(data2fit,kk++) = 0.16431925;
            break;
        case 5:
            // Shinoda 2007 (mikro gramm / milli Liter): nData=9; nAPAP_G_S=3;
            // 1000mg oral. meanBW = 60
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0;       Ith(data2fit,kk++) = 0.0;         Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.25     ; Ith(data2fit,kk++) = 16.379310; Ith(data2fit,kk++) = 1.6091954 ; Ith(data2fit,kk++) = 1.6091954;
            Ith(data2fit,kk++) = 0.5 ;  Ith(data2fit,kk++) = 17.356322; Ith(data2fit,kk++) = 4.7701149 ; Ith(data2fit,kk++) = 3.5632184;
            Ith(data2fit,kk++) = 0.75     ; Ith(data2fit,kk++) = 15.459770; Ith(data2fit,kk++) = 6.2643678; Ith(data2fit,kk++) = 3.8505747;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 13.850575; Ith(data2fit,kk++) = 7.3563218; Ith(data2fit,kk++) = 3.8505747;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 10.632184; Ith(data2fit,kk++) = 10.3448276; Ith(data2fit,kk++) = 3.9655172;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 8.448276; Ith(data2fit,kk++) = 9.7126437; Ith(data2fit,kk++) = 3.4482759;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 6.781609; Ith(data2fit,kk++) = 8.4482759; Ith(data2fit,kk++) = 2.9885057;
            Ith(data2fit,kk++) = 6.0 ;  Ith(data2fit,kk++) = 4.712644; Ith(data2fit,kk++) = 5.5172414; Ith(data2fit,kk++) = 1.9540230;
            break;
        case 6:
            // Tan 2012 (mikro gramm / milli Liter): nData=15; nAPAP_G_S=2;
            // 500mg oral meanBW = 62.5 [5.6], meanAge = 23.7 [1.8]
            Ith(data2fit,kk++) = 0.0     ;                 Ith(data2fit,kk++) = 0.0;       Ith(data2fit,kk++) = 0.0;         // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.166666666666667     ;   Ith(data2fit,kk++) = 4.20930233; Ith(data2fit,kk++) = 0.27906977;  // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.333333333333333 ;     Ith(data2fit,kk++) = 6.23255814; Ith(data2fit,kk++) = 1.13953488;  // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.5     ;                 Ith(data2fit,kk++) = 6.69767442; Ith(data2fit,kk++) = 1.97674419;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.75     ;                 Ith(data2fit,kk++) = 7.18604651; Ith(data2fit,kk++) = 3.39534884;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.0     ;                 Ith(data2fit,kk++) = 6.55813953; Ith(data2fit,kk++) = 4.48837209;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.5     ;                 Ith(data2fit,kk++) = 5.83720930; Ith(data2fit,kk++) = 4.76744186;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 2.0     ;                 Ith(data2fit,kk++) = 5.46511628; Ith(data2fit,kk++) = 5.11627907;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 3.0     ;                 Ith(data2fit,kk++) = 4.11627907; Ith(data2fit,kk++) = 6.16279070;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 4.0     ;                 Ith(data2fit,kk++) = 3.27906977; Ith(data2fit,kk++) = 5.48837209;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 6.0 ;                   Ith(data2fit,kk++) = 1.69767442; Ith(data2fit,kk++) = 3.67441860;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 8.0 ;                   Ith(data2fit,kk++) = 0.95348837; Ith(data2fit,kk++) = 2.20930233;   // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 10.0 ;                  Ith(data2fit,kk++) = 0.60952381; Ith(data2fit,kk++) = 1.60000000; // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 12.0 ;                  Ith(data2fit,kk++) = 0.37142857; Ith(data2fit,kk++) = 0.92380952; // Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 24.0 ;                  Ith(data2fit,kk++) = 0.05714286; Ith(data2fit,kk++) = 0.14285714; // Ith(data2fit,kk++) = 0.0;
            break;
        case 7:
            // Volak 2013 (mikro gramm / milli Liter): nData=11; nAPAP_G_S=3;
            // 325mg oral meanBW =  (range:61-93), meanAge = (range:24-52)
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0;       Ith(data2fit,kk++) = 0.0;         Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 1.98076923; Ith(data2fit,kk++) = 0.3538462;   Ith(data2fit,kk++) = 0.72318339;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 2.70604396; Ith(data2fit,kk++) = 2.169231;   Ith(data2fit,kk++) = 1.56747405;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 3.09340659; Ith(data2fit,kk++) = 3.492308;   Ith(data2fit,kk++) = 2.23183391;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 2.83791209; Ith(data2fit,kk++) = 4.323077;   Ith(data2fit,kk++) = 2.33564014;
            Ith(data2fit,kk++) = 2.5     ; Ith(data2fit,kk++) = 2.35989011; Ith(data2fit,kk++) = 4.769231;   Ith(data2fit,kk++) = 2.22491349;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 2.00549451; Ith(data2fit,kk++) = 5.030769;   Ith(data2fit,kk++) = 1.99653979;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 1.56043956; Ith(data2fit,kk++) = 4.630769;   Ith(data2fit,kk++) = 1.71280277;
            Ith(data2fit,kk++) = 6.0     ; Ith(data2fit,kk++) = 0.82692308; Ith(data2fit,kk++) = 3.153846;   Ith(data2fit,kk++) = 1.04152249;
            Ith(data2fit,kk++) = 8.0 ;   Ith(data2fit,kk++) = 0.50549451; Ith(data2fit,kk++) = 1.923077;   Ith(data2fit,kk++) = 0.66089965;
            Ith(data2fit,kk++) = 10.0 ;   Ith(data2fit,kk++) = 0.35714286; Ith(data2fit,kk++) = 1.323077;   Ith(data2fit,kk++) = 0.50865052;
            break;
        case 8:
            // Yin 2001 (mikro gramm / milli Liter): nData=12; nAPAP_G_S=1;
            // 500mg oral meanBW = 62.7 ± 5.6, meanAge = 24.1 ± 7.1, meanHeight = 170.6 ± 4.4
            Ith(data2fit,kk++) = 0.0     ; Ith(data2fit,kk++) = 0.0;       // Ith(data2fit,kk++) = 0.0;         Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.25     ; Ith(data2fit,kk++) = 3.0861244; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.5     ; Ith(data2fit,kk++) = 5.7416268; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.75     ; Ith(data2fit,kk++) = 5.9569378; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.0     ; Ith(data2fit,kk++) = 6.2200957; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.5     ; Ith(data2fit,kk++) = 5.4545455; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 2.0     ; Ith(data2fit,kk++) = 4.9282297; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 3.0     ; Ith(data2fit,kk++) = 3.9712919; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 4.0     ; Ith(data2fit,kk++) = 3.0143541; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 6.0 ;   Ith(data2fit,kk++) = 1.7224880; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 8.0 ;   Ith(data2fit,kk++) = 1.0526316; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 10.0 ;   Ith(data2fit,kk++) = 0.7177033; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            break;
        case 9:
            // Zhu 2007 (mikro gramm / milli Liter): nData=14; nAPAP_G_S=1;
            // 650mg oral meanBW = 60 (Chinese)
            Ith(data2fit,kk++) = 0.0     ;               Ith(data2fit,kk++) = 0.0;       // Ith(data2fit,kk++) = 0.0;         Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.166666666666667     ; Ith(data2fit,kk++) = 6.4959155; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.333333333333333     ; Ith(data2fit,kk++) = 7.9666206; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.50     ;               Ith(data2fit,kk++) = 7.9346487; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 0.75     ;               Ith(data2fit,kk++) = 6.5278874; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.0     ;               Ith(data2fit,kk++) = 6.7836622; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 1.5     ;               Ith(data2fit,kk++) = 5.2490134; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 2.0     ;               Ith(data2fit,kk++) = 4.4816891; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 3.0     ;               Ith(data2fit,kk++) = 2.9470403; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 4.0     ;               Ith(data2fit,kk++) = 2.4035189; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 6.0 ;                 Ith(data2fit,kk++) = 1.1885887; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 10.0 ;                Ith(data2fit,kk++) = 0.4532361; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 15.0 ;                Ith(data2fit,kk++) = 0.2614051; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            Ith(data2fit,kk++) = 24.0 ;                Ith(data2fit,kk++) = 0.1335177; // Ith(data2fit,kk++) = 0.0;   Ith(data2fit,kk++) = 0.0;
            break;
        default: return(1); break;
    }
    return(0);
}

int getLogLikelihood(int nData, int nAPAP_G_S, double *logLikelihood_can, N_Vector odeSolution, N_Vector data2fit, realtype sigmaLL, int LogNormalBool){
    // additives Fehlermodell: Y = f() + epsilon bzw. log-trafo: ln(Y) = ln(f()) + epsilon, mit epsilon ~ N(0,sigma^2)
    //    --> Y ~ logN(log(f()), sigma), weil log(Y) ~ N(log(f()), sigma^2), nach Zurlinden,2017
    // ich folge hier nun: https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html#statistical-model-prior-knowledge-and-unexplained-variation
    //    --> data2fit ~ logN(log(odeSolution) ,sigma^2)

    // 'normal': denom = 2.*sigma.^2; LogLikeli = -0.5*N.*log(pi.*denom) - sum((z - data).^2)./denom ;
    // 'log': denom = 2.*sigma.^2; LogLikeli = sum( -0.5*log(pi.*denom.*x.^2) - ((log(z) - data).^2)./denom );

    int iout, kk;
    realtype logLikelihood_can_temp;

    iout = 2;                           // iout für die Datenzeitpunkte verwenden
    logLikelihood_can_temp = 0.0;
    *logLikelihood_can = 0.0;

    // sigmaLL = 0.5; // wie wählen??-> nach Hsieh ea.,2018,S.9: jeweils ein sigma für APAP/-G/-S (weil die letztlich seperat gemessen wurden?!). Die LogLikeli-Funktion bei Hsieh geht über alle N Datenpunkte, d.h. APAP/-G/-S sind alle mit dabei (?!).

    for (kk = nAPAP_G_S+1; kk <= nData*nAPAP_G_S; kk++) {     //bei kk=2 starten, weil bei kk=1 ich noch bei t=0 bin; also alle Werte = 0.0 --> log(0.0)=nan

        if (LogNormalBool) {
            *logLikelihood_can = *logLikelihood_can + pow(log(Ith(odeSolution,kk)) - log(Ith(data2fit,iout+kk)),2);
            logLikelihood_can_temp += log(Ith(data2fit,iout+kk)) ;
        } else {
            *logLikelihood_can += pow(Ith(odeSolution,kk) - Ith(data2fit,iout+kk),2);
        }
        if ((kk % nAPAP_G_S)==0 ) iout++;
    }

    *logLikelihood_can = *logLikelihood_can / (2*pow(sigmaLL,2));
    *logLikelihood_can = - *logLikelihood_can - logLikelihood_can_temp - nData*nAPAP_G_S*log(sigmaLL) - (nData*nAPAP_G_S*log(2*M_PI))/2;

    return(0);
}

int getLogLikelihoodTheta(int nData, int nAPAP_G_S, double *logLikelihood_can, double *LikelihoodNumerator, N_Vector odeSolution, N_Vector data2fit, realtype sigmaLL, int LogNormalBool){
    // additives Fehlermodell: Y = f() + epsilon bzw. log-trafo: ln(Y) = ln(f()) + epsilon, mit epsilon ~ N(0,sigma^2)
    //    --> Y ~ logN(log(f()), sigma), weil log(Y) ~ N(log(f()), sigma^2), nach Zurlinden,2017
    // ich folge hier nun: https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html#statistical-model-prior-knowledge-and-unexplained-variation
    //    --> data2fit ~ logN(log(odeSolution) ,sigma^2)

    // 'normal': denom = 2.*sigma.^2; LogLikeli = -0.5*N.*log(pi.*denom) - sum((z - data).^2)./denom ;
    // 'log': denom = 2.*sigma.^2; LogLikeli = sum( -0.5*log(pi.*denom.*x.^2) - ((log(z) - data).^2)./denom );

    int iout, kk, colsODESolu=3, idx;
    realtype logLikelihood_can_temp;

    iout = 2;                           // iout für die Datenzeitpunkte verwenden
    logLikelihood_can_temp = 0.0;
    *logLikelihood_can = 0.0;
    *LikelihoodNumerator = 0.0;
    // sigmaLL = 0.5; // wie wählen??-> nach Hsieh ea.,2018,S.9: jeweils ein sigma für APAP/-G/-S (weil die letztlich seperat gemessen wurden?!). Die LogLikeli-Funktion bei Hsieh geht über alle N Datenpunkte, d.h. APAP/-G/-S sind alle mit dabei (?!).

    // LogLikelihood berechnen:
    idx = colsODESolu+1;
    for (kk = nAPAP_G_S+1; kk <= nData*nAPAP_G_S; kk++) {     //bei kk=2 starten, weil bei kk=1 ich noch bei t=0 bin; also alle Werte = 0.0 --> log(0.0)=nan // aber ich muss den Vorfaktor von der Normalverteilung trotzdem von t=0 mitreinnehmen!!
        if (LogNormalBool) {
            *LikelihoodNumerator += pow(log(Ith(odeSolution,idx)) - log(Ith(data2fit,iout+kk)),2); // für den sigmaLL-Block extra abspeichern
            logLikelihood_can_temp += log(Ith(data2fit,iout+kk)) ;//0.5*(nData*nAPAP_G_S)*log(2*M_PI*pow(sigmaLL,2)*pow(Ith(data2fit,iout+kk),2));
        } else {
            *LikelihoodNumerator += pow(Ith(odeSolution,idx) - Ith(data2fit,iout+kk),2);
        }
        if ((kk % nAPAP_G_S)==0 ) {
            iout++;
            idx += colsODESolu-nAPAP_G_S;
        }
        idx++;
    }

    *logLikelihood_can = *LikelihoodNumerator / (2*pow(sigmaLL,2)); // sigma hier gibt jetzt die Datenunsicherheit an
    *logLikelihood_can = - *logLikelihood_can - logLikelihood_can_temp - nData*nAPAP_G_S*log(sigmaLL) - (nData*nAPAP_G_S*log(2*M_PI))/2;

    return(0);
}

int getLogLikelihoodSigma(int nData, int nAPAP_G_S, double *logLikelihood_can, N_Vector LikelihoodNumerator, int block, N_Vector data2fit, realtype sigmaLL, int LogNormalBool){
    // additives Fehlermodell: Y = f() + epsilon bzw. log-trafo: ln(Y) = ln(f()) + epsilon, mit epsilon ~ N(0,sigma^2)
    //    --> Y ~ logN(log(f()), sigma), weil log(Y) ~ N(log(f()), sigma^2), nach Zurlinden,2017
    // bzgl. LogNormal-Verteilung, siehe: https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html#statistical-model-prior-knowledge-and-unexplained-variation
    //    --> data2fit ~ logN(log(odeSolution) ,sigma^2)

    // 'normal': f(x) = 1/(2*pi*sigma^2)^(1/2) * exp(- (x-mu)^2 / (2*sigma^2) )
    // 'log': f(x) = 1/((2*pi)^(1/2)*sigma*x) * exp(- (log(x)-mu)^2 / (2*sigma^2) )

    int iout, kk;
    realtype logLikelihood_can_temp;

    iout = 2;                           // iout für die Datenzeitpunkte verwenden
    logLikelihood_can_temp = 0.0;
    *logLikelihood_can = 0.0;
    // sigmaLL = 0.5; // wie wählen??-> nach Hsieh ea.,2018,S.9: jeweils ein sigma für APAP/-G/-S (weil die letztlich seperat gemessen wurden?!). Die LogLikeli-Funktion bei Hsieh geht über alle N Datenpunkte, d.h. APAP/-G/-S sind alle mit dabei (?!).

    // LogLikelihood berechnen:
    for (kk = nAPAP_G_S+1; kk <= nData*nAPAP_G_S; kk++) {     //bei kk=2 starten, weil bei kk=1 ich noch bei t=0 bin; also alle Werte = 0.0 --> log(0.0)=nan
        if (LogNormalBool) {
            logLikelihood_can_temp += log(Ith(data2fit,iout+kk)) ;//0.5*(nData*nAPAP_G_S)*log(2*M_PI*pow(sigmaLL,2)*pow(Ith(data2fit,iout+kk),2));
        }
        if ((kk % nAPAP_G_S)==0 ) iout++;
    }
    *logLikelihood_can = Ith(LikelihoodNumerator,block) / (2*pow(sigmaLL,2)); // sigma hier gibt jetzt die Datenunsicherheit an
    *logLikelihood_can = - *logLikelihood_can - logLikelihood_can_temp - nData*nAPAP_G_S*log(sigmaLL) - (nData*nAPAP_G_S*log(2*M_PI))/2;

    return(0);
}

int getLogPrior(double *logPrior_can, N_Vector thetaCan){
    /* Zurlinden, Tabelle 5, S.274, enthält die Priorverteilungen für die Parameter.
     *
     * Zurlinden, 2017, S. 144; Hsieh, 2018, S.3: distributions for all parameter priors were log-transformed and sampled either as normal or uniform distributions.; parameters assumed to be independently distributed --> sampled independently.
     * d.h. wenn ich die Parameterkandidaten mit Normal-/Uniformverteilungen wie unten auswerte, gehe ich davon aus, dass ich
     * die Kandidaten logarithmiert habe. (ist jetzt so implementiert)
     *
     * Bei den Normalverteilungen N(a,b) ist a der Mittelwert und b der Variationskoeffizient.
     * Variationskoeffizient = Standardabweichung / Mittelwert = sigma / mu .
     *
     * Bei den Gleichverteilungen U(a,b) sind a und b die Maximal- bzw. Minimal-
     * werte.
     */
    int ctr = 1;
    realtype mu, cv, sigma, max, min, thetaCanI;

    *logPrior_can = 0.0;

    /* Acetaminophen absorption */
    // T_G: N(0.23, 0.5) -> sigma = 0.5 * 0.23 = 0.115
    mu    = log(0.23);
    cv    = 0.23;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // T_P: N(0.033, 0.5) -> sigma = 0.5 * 0.033 = 0.0165
    mu    = log(0.033);
    cv    = 0.033;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));

    /* Phase I metabolism */
    // K_APAP_Mcyp: N(130, 1) -> sigma = 1 * 130 = 130
    mu    = log(130.0);
    cv    = 1.0;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // V_MCcyp: U(0.14, 2900)
    max = 2900.0;
    min = 0.14;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

    /* Phase II metabolism: sulfation */
    // K_APAP_Msult: N(300,1) -> sigma = 1 * 300 = 300
    mu    = log(300.0);
    cv    = 1.0;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // K_APAP_Isult: N(526,1) -> sigma = 1 * 526 = 526
    mu    = log(526.0);
    cv    = 1.0;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // K_PAPS_Msult: N(0.5,0.5) -> sigma = 0.5 * 0.5 = 0.25
    mu    = log(0.5);
    cv    = 0.5;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // V_MCsult: U(1,3.26E6)
    max = 3.26e+6;
    min = 1.0;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

    /* Phase I metabolism: glucuronidation */
    // K_APAP_Mugt: N(6.0E4,1) -> sigma = 1 * 6.0E4 = 6.0E4
    mu    = log(6.0e+4);
    cv    = 1.0;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // K_APAP_Iugt: N(5.8E4,0.25) -> sigma = 0.25 * 5.8E4 = 1.45E4
    mu    = log(5.8e+4);
    cv    = 0.25;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // K_UDPGA_Mugt: N(0.5,0.5) -> sigma = 0.25
    mu    = log(0.5);
    cv    = 0.5;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // V_MCugt: U(1,3.26E6)
    max = 3.26e+6;
    min = 1.0;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

    /* Active hepatic transporters */
    // K_APAPG_Mmem: N(1.99E4, 0.3) -> sigma = 0.3 * 1.99E4 = 5.97E3
    mu    = log(1.99e+4);
    cv    = 0.3;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // V_APAPG_Mmem: U(1.09E3,3.26E6)
    max = 1.09e+3;
    min = 3.26e+6;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }
    // K_APAPS_Mmem: N(2.29E4, 0.22) -> sigma = 0.22 * 2.29E4 = 5.038E3
    mu    = log(2.29e+4);
    cv    = 0.22;
    sigma = mu * cv;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    // printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2));
    *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
    // V_APAPS_Mmem: U(1.09E3,3.26E6)
    max = 1.09e+3;
    min = 3.26e+6;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

    /* Cofactor synthesis */
    // k_synUDPGA: U(1,4.43E5)
    max = 4.43e+5;
    min = 1.0;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }
    // k_synPAPS: U(1,4.43E5)
    max = 4.43e+5;
    min = 1.0;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

    /* Clearance */
    // k_APAP_R0: U(2.48E-3,2.718)
    max = 2.718;
    min = 2.48e-3;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }
    // k_APAPG_R0: U(2.48E-3,2.718)
    max = 2.718;
    min = 2.48e-3;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }
    // k_APAPS_R0: U(2.48E-3,2.718)
    max = 2.718;
    min = 2.48e-3;
    thetaCanI = Ith(thetaCan,ctr++); // printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
        // printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min)));
        *logPrior_can += -log(log(max)-log(min));
    }
    else {
        /* logPrior_can = logPrior_can - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
    }

//
//    /* Acetaminophen absorption */
//    // T_G: N(0.23, 0.5) -> sigma = 0.5 * 0.23 = 0.115
//    mu = 0.23;
//    sigma = 0.115;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // T_P: N(0.033, 0.5) -> sigma = 0.5 * 0.033 = 0.0165
//    mu = 0.033;
//    sigma = 0.0165;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//
//    /* Phase I metabolism */
//    // K_APAP_Mcyp: N(130, 1) -> sigma = 1 * 130 = 130
//    mu = 130.0;
//    sigma = 130.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // V_MCcyp: U(0.14, 2900)
//    max = 2900.0;
//    min = 0.14;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
//    /* Phase II metabolism: sulfation */
//    // K_APAP_Msult: N(300,1) -> sigma = 1 * 300 = 300
//    mu    = 300.0;
//    sigma = 300.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // K_APAP_Isult: N(526,1) -> sigma = 1 * 526 = 526
//    mu    = 526.0;
//    sigma = 526.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // K_PAPS_Msult: N(0.5,0.5) -> sigma = 0.5 * 0.5 = 0.25
//    mu    = 0.5;
//    sigma = 0.25;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // V_MCsult: U(1,3.26E6)
//    max = 3.26e+6;
//    min = 1.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
//    /* Phase I metabolism: glucuronidation */
//    // K_APAP_Mugt: N(6.0E4,1) -> sigma = 1 * 6.0E4 = 6.0E4
//    mu    = 6.0e+4;
//    sigma = 6.0e+4;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // K_APAP_Iugt: N(5.8E4,0.25) -> sigma = 0.25 * 5.8E4 = 1.45E4
//    mu    = 5.8e+4;
//    sigma = 1.45e+4;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // K_UDPGA_Mugt: N(0.5,0.5) -> sigma = 0.25
//    mu    = 0.5;
//    sigma = 0.25;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // V_MCugt: U(1,3.26E6)
//    max = 3.26e+6;
//    min = 1.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
//    /* Active hepatic transporters */
//    // K_APAPG_Mmem: N(1.99E4, 0.3) -> sigma = 0.3 * 1.99E4 = 5.97E3
//    mu    = 1.99e+4;
//    sigma = 5.97e+3;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // V_APAPG_Mmem: U(1.09E3,3.26E6)
//    max = 1.09e+3;
//    min = 3.26e+6;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//    // K_APAPS_Mmem: N(2.29E4, 0.22) -> sigma = 0.22 * 2.29E4 = 5.038E3
//    mu    = 2.29e+4;
//    sigma = 5.038e+3;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    printf("*logPrior_can: %g += %g\n", *logPrior_can, - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)); *logPrior_can += - 0.5*log(2*M_PI*pow(sigma,2)) - pow(mu-thetaCanI,2)/(2*pow(sigma,2));
//    // V_APAPS_Mmem: U(1.09E3,3.26E6)
//    max = 1.09e+3;
//    min = 3.26e+6;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
//    /* Cofactor synthesis */
//    // k_synUDPGA: U(1,4.43E5)
//    max = 4.43e+5;
//    min = 1.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//    // k_synPAPS: U(1,4.43E5)
//    max = 4.43e+5;
//    min = 1.0;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
//    /* Clearance */
//    // k_APAP_R0: U(2.48E-3,2.718)
//    max = 2.718;
//    min = 2.48e-3;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//    // k_APAPG_R0: U(2.48E-3,2.718)
//    max = 2.718;
//    min = 2.48e-3;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* *logPrior_can += - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//    // k_APAPS_R0: U(2.48E-3,2.718)
//    max = 2.718;
//    min = 2.48e-3;
//    thetaCanI = Ith(thetaCan,ctr++); printf("thetaCanI = Ith(thetaCan,%d) = %g\n", ctr, thetaCanI);
//    if (thetaCanI<=log(max) && thetaCanI>=log(min)) {
//        printf("*logPrior: %g += %g\n", *logPrior_can ,-log(log(max)-log(min))); *logPrior_can += -log(log(max)-log(min));
//    }
//    else {
//        /* logPrior_can = logPrior_can - log(0); // nicht gut...->abbrechen? Kandidaten verwerfen? */
//    }
//
    return(0);
} // getLogPrior

/******************************************************************************/
/************************** functions for swap ********************************/
/******************************************************************************/
long getSeed(){
    /* calculates and returns the current time
     */
    struct timeval timevalue;

    gettimeofday(&timevalue, 0);
    return(timevalue.tv_sec + timevalue.tv_usec);
}

int getSwapscheme(int swaps_max, int swaps, gsl_rng* gslrng, int* swapscheme){
    int nextswap[2], processors[swaps_max], i, kk, allocated, ctr, breaker[2];
    // printf("processors=[");
    for (i = 0; i < swaps_max; i++) {
        //        if (i<swaps) {
        swapscheme[i]=i;
        //        } else {
        //            swapscheme[i]=-1;
        //        }
        processors[i] = 0;
        // printf(" %2d", processors[i]);
    } //printf("]\n");

    // printf("swapscheme=[");
    // for (i = 0; i < swaps; i++) {
    //   printf(" %2d", swapscheme[i]);
    // }
    // printf("]\n");

    /* Tauschschema ermitteln
     *
     * Idee:
     */
    allocated = 0;
    for (kk = 0; kk < swaps/2; kk++) {
        // würfele eine Zahl, die die relative Position des nächsten Tauschpartners anzeigt.
        nextswap[0] = gsl_rng_uniform_int(gslrng, swaps_max-allocated); // printf("swaps-alloc %d\n",swaps_max-allocated);
        allocated++;

        nextswap[1] = gsl_rng_uniform_int(gslrng, swaps_max-allocated); // printf("swaps-alloc %d\n",swaps_max-allocated); // nächste relative Position als Tauschposition
        allocated++;
//        printf("ns0 = %d\n", nextswap[0]);
//        printf("ns1 = %d\n", nextswap[1]);
        if (nextswap[1]>=nextswap[0]) { // prüfe die Lage der relativen Positionen zueinander
            nextswap[1]++;
        }
//        printf("ns1 = %d\n", nextswap[1]);

        ctr = -1;
        breaker[0] = 0; breaker[1] = 0;
        for (i=0; i < swaps_max; i++) {
            if (processors[i]==0)   { ctr++; }
//            printf("i=%d ctr=%d\n",i,ctr);
            if ((ctr==nextswap[0])&&(breaker[0]==0))    {nextswap[0]=i; breaker[0]=1; }
            if ((ctr==nextswap[1])&&(breaker[1]==0))    {nextswap[1]=i; breaker[1]=1; }
            if ((breaker[0]==1)&&(breaker[1]==1))       { break; }
        }

        processors[nextswap[0]] = 1;
        processors[nextswap[1]] = 1;

        swapscheme[nextswap[1]] = nextswap[0];
        swapscheme[nextswap[0]] = nextswap[1];

//         printf("swapscheme=[");
//         for (i = 0; i < swaps_max; i++) {
//           printf(" %2d", swapscheme[i]);
//         }
//         printf("]\t");
//         printf("processors=[");
//         for (i = 0; i < swaps_max; i++) {
//           printf(" %2d", processors[i]);
//         }
//         printf("]\n");
    }
    return(0);
}
