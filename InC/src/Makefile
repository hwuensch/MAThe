# -*- mode: makefile -*-

SHELL = sh

ODIR = obj
LDIR = ../lib
IDIR = ../include

prefix       = /usr/local
exec_prefix  = /usr/local
includedir   = /usr/local/include
libdir       = /usr/local/lib

CPP      = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
CPPFLAGS = -O3 -DNDEBUG
CC       = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
CFLAGS   = -O3 -DNDEBUG -I.
LDFLAGS  = 
LIBS     =  -lm -lsprng -lgmp -lgsl -lgslcblas -lmpi

LINKFLAGS = -Wl,-rpath,/usr/local/lib

# ----------------------------------------------------------------------------------------

LIBRARIES_LAPACK = -lsundials_sunlinsollapackdense -lsundials_sunlinsollapackband  
LINKFLAGS_LAPACK = ${LINKFLAGS}

INCLUDES_SPRNG = /Users/hendrik_w/Documents/Studium/2Köln/Master/5.Sem-WS1819/Bayer/Code/Heimarbeit/sprng2.0/include
LIBRARIES_SPRNG = /Users/hendrik_w/Documents/Studium/2Köln/Master/5.Sem-WS1819/Bayer/Code/Heimarbeit/sprng2.0/lib

TMP_INCS  = ${includedir} ${INCLUDES_SPRNG} ${IDIR}
INCLUDES  = $(addprefix -I, ${TMP_INCS})
TMP_LIBRS = ${libdir} ${LIBRARIES_SPRNG} ${LDIR}
LIB_DIR   = $(addprefix -L, ${TMP_LIBRS})
LIBRARIES = -lsundials_cvode -lsundials_nvecserial ${LIBS}

# ----------------------------------------------------------------------------------------

EXAMPLES = pbpkmcmc_seriellMH

DEPENDENCIES_C 	= pbpkmcmcfunc
_DEPS_H 	= pbpkmcmc.h
DEPS_H		= $(patsubst %,$(IDIR)/%,$(_DEPS_H))

_OBJECTS 		= ${EXAMPLES:=.o}
OBJECTS			= $(patsubst %,$(ODIR)/%,$(_OBJECTS))
_OBJECTS_DEPENDENCIES_C	= ${DEPENDENCIES_C:=.o}
OBJECTS_DEPENDENCIES_C 	= $(patsubst %,$(ODIR)/%,$(_OBJECTS_DEPENDENCIES_C))

# ----------------------------------------------------------------------------------------

#.SUFFIXES : .o .c

#.c.o : $(DEPS_H)
#	${CC} ${CPPFLAGS} ${CFLAGS} ${INCLUDES}  -c $<

$(ODIR)/%.o: %.c $(DEPS_H)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDES)

# ----------------------------------------------------------------------------------------

all: MH blockMH blockmultMH
run: MH_run blockMH_run blockmultMH_run
run_chain: MH_run blockMH_run

#all: seriellMH seriellblockMH parallelblockMH parallelblockmultipleMH sPHS_blockMH sPHS_MH
#run: seriellMH_run seriellblockMH_run parallelblockMH_run parallelblockmultipleMH_run sPHS_blockMH_run sPHS_MH_run

seriell: seriellMH_run seriellblockMH_run
parallel: parallelblockMH_run parallelblockmultipleMH_run
sPHS: sPHS_MH_run sPHS_blockMH_run

# ----------------------------------------------------------------------------------------

MH_run: MH
#	swapsStart = $(swapsStart); \
#	while [ ${swapsStart} -le $(swaps) ] ; do \
#		echo "mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) $(swapsStart)" ; \
#		mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) $(swapsStart) ; \
#		swapsStart = $(shell expr $(swapsStart) + $(nAdd) ) ; \
#	done; \
#	true
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 0
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 2
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 4
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 6
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 8
#	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) 10
	mpirun --oversubscribe -n $(nProc) ./MH $(iterAll) $(swaps) $(seed)
MH: obj/pbpkmcmc_MH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

# ----------------------------------------------------------------------------------------

blockMH_run: blockMH
	mpirun --oversubscribe -n $(nProc) ./blockMH $(iterAll) $(swaps) $(seed)
blockMH: obj/pbpkmcmc_blockMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

# ----------------------------------------------------------------------------------------

blockmultMH_run: blockmultMH
	mpirun --oversubscribe -n $(nProc) ./blockmultMH $(iterAll) $(swaps) $(seed) $(mChainsBool) $(bPR)
blockmultMH: obj/pbpkmcmc_blockmultMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

# ----------------------------------------------------------------------------------------

seriellMH_run: seriellMH
	mpirun --oversubscribe -n 1 ./seriellMH $(iterAll)
seriellMH: obj/pbpkmcmc_seriellMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

seriellblockMH_run: seriellblockMH
	mpirun --oversubscribe -n 1 ./seriellblockMH $(iterAll)
seriellblockMH: obj/pbpkmcmc_seriellblockMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}
	
parallelblockmultipleMH_run: parallelblockmultipleMH
	mpirun --oversubscribe -n $(nProc) ./parallelblockmultipleMH $(iterAll)
parallelblockmultipleMH: obj/pbpkmcmc_parallelblockmultipleMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

parallelblockMH_run: parallelblockMH
	mpirun --oversubscribe -n 11 ./parallelblockMH $(iterAll)
parallelblockMH: obj/pbpkmcmc_parallelblockMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

sPHS_MH_run: sPHS_MH
	mpirun --oversubscribe -n $(nProc) ./sPHS_MH $(iterAll) $(swapBool)
sPHS_MH: obj/pbpkmcmc_sPHS_MH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

sPHS_blockMH_run: sPHS_blockMH
	mpirun --oversubscribe -n $(nProc) ./sPHS_blockMH $(iterAll) $(swapBool)
sPHS_blockMH: obj/pbpkmcmc_sPHS_blockMH.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}


examples: ${OBJECTS}
	@for i in ${EXAMPLES} ; do \
	  echo "${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES_C} ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}" ; \
	  ${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES_C} ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS} ; \
	done

${OBJECTS}: ${OBJECTS_DEPENDENCIES_C}

clean:
	rm -f ${OBJECTS_DEPENDENCIES_C}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}

# ----------------------------------------------------------------------------------------
