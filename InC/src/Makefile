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
LIBRARIES = ${LIBS}

# ----------------------------------------------------------------------------------------

EXAMPLES = mcmcNormal

DEPENDENCIES_C 	= mcmcNormalfunc
_DEPS_H 	= mcmcNormal.h
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

all: mcmcNormalMain
run: mcmcNormal_run

# ----------------------------------------------------------------------------------------

mcmcNormal_run: mcmcNormalMain
	mpirun --oversubscribe -n $(nProc) ./mcmcNormalMain $(iterAll) $(dim) $(start) # $(swaps) $(seed) $(mChainsBool) $(bPR)
mcmcNormalMain: obj/mcmcNormalMain.o ${OBJECTS_DEPENDENCIES_C}
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS} ${INCLUDES} ${LIB_DIR} ${LIBRARIES} ${LINKFLAGS}

# ----------------------------------------------------------------------------------------

${OBJECTS}: ${OBJECTS_DEPENDENCIES_C}

clean:
	rm -f ${OBJECTS_DEPENDENCIES_C}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}

# ----------------------------------------------------------------------------------------
