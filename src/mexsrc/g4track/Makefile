# g4track code - for interfacing GEANT4 tracking with Lucretia
# "make all" makes shared obj library for use when building Lucretia
# mex files
# "make clean" to clean up all object and library files
#
# =============================================================================
# ASSUMES GEANT4 (VERSION 10+) INSTALLED IN STANDARD WAY WITH PATH DEFINED
# IN SYSTEM ("geant4-config" script must run) AND ALL MATERIAL FILES INCLUDED
# =============================================================================
#
# GW: May 29, 2014

 CCOMP := mex -largeArrayDims CC=c++ 

# CFLAGS := CXXFLAGS='$$CXXFLAGS $(shell geant4-config --cflags)' -DG4MULTITHREADED
 CLIBS := $(shell geant4-config --libs)
 CFLAGS := CXXFLAGS='$$CXXFLAGS $(shell geant4-config --cflags)'

# g4track Library
all : g4track.o PhysicsList.o lucretiaManager.o geomConstruction.o actionInitialization.o primaryGeneratorAction.o runAction.o trackingAction.o FieldSetup.o GlobalField.o lSession.o steppingAction.o StackingAction.cpp

# cleanup
clean::
	-rm -f *.o
	-rm -f *.so*
	-rm -f libg4track.a

g4track.o : ../LucretiaCommon.h g4track.h geomConstruction.hh actionInitialization.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c g4track.cpp
	ar -ru libg4track.a g4track.o

lSession.o : lSession.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c lSession.cpp
	ar -ru libg4track.a lSession.o

PhysicsList.o: PhysicsList.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c PhysicsList.cpp
	ar -ru libg4track.a PhysicsList.o

geomConstruction.o: geomConstruction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c geomConstruction.cpp
	ar -ru libg4track.a geomConstruction.o

actionInitialization.o: actionInitialization.hh primaryGeneratorAction.hh runAction.hh eventAction.hh trackingAction.hh stackingAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c actionInitialization.cpp
	ar -ru libg4track.a actionInitialization.o

eventAction.o: eventAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c eventAction.cpp
	ar -ru libg4track.a eventAction.o

steppingAction.o: steppingAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c steppingAction.cpp
	ar -ru libg4track.a steppingAction.o

primaryGeneratorAction.o: primaryGeneratorAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c primaryGeneratorAction.cpp
	ar -ru libg4track.a primaryGeneratorAction.o

runAction.o: runAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c runAction.cpp
	ar -ru libg4track.a runAction.o

trackingAction.o: trackingAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c trackingAction.cpp
	ar -ru libg4track.a trackingAction.o

stackingAction.o: stackingAction.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c stackingAction.cpp
	ar -ru libg4track.a stackingAction.o

lucretiaManager.o: lucretiaManager.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c lucretiaManager.cpp
	ar -ru libg4track.a lucretiaManager.o

FieldSetup.o: FieldSetup.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c FieldSetup.cpp
	ar -ru libg4track.a FieldSetup.o

GlobalField.o: GlobalField.hh
	$(CCOMP) $(CFLAGS) $(CLIBS) -c GlobalField.cpp
	ar -ru libg4track.a GlobalField.o
