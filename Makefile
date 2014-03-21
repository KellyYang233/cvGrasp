#
# K3Engine Makefile
#
# Copyright (c) 2009
# Pyry Matikainen
# Prasanna Velagapudi
#
# (Adapted from:
#  http://wiki.osdev.org/Makefile
#  https://negix.net/trac/pdclib/browser/trunk/Makefile)
#
# 'make depend' uses makedepend to automatically generate dependencies 
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'k3engine'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC := g++
C_REGULAR_NOT_CPP_BECAUSE_THE_ABOVE_IS_NAMED_CC := cc

# define any compile-time flags
CFLAGS := -g -Wall

# define any directories containing header files other than /usr/include
INCLUDES := -I. -I/usr/local/include/
#INCLUDES := -I. -I/IUS/vmr101/software/ubuntu10.04/opencv-2.4.2/include

# define library paths in addition to /usr/lib
#LFLAGS := -L/vmr/...
LFLAGS :=  -L. -Wl,-rpath=/usr/local/lib/ -L/usr/local/lib/

# define any libraries to link into executable:
LIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lopencv_nonfree

# define project directories
PROJDIRS := .

# define the output staging directory structure
BUILDDIR := .

# define the CXX source files
#CXXSRC := $(shell find $(PROJDIRS) -mindepth 1 -maxdepth 3 -name "*.cpp")
#CSRC := $(shell find $(PROJDIRS) -mindepth 1 -maxdepth 3 -name "*.c") 

CXXSRC := commonUse.cpp Skeleton.cpp cvGrasp.cpp DataPreparation.cpp FeatureExtractor.cpp HiCluster.cpp

# define the C object files 
OBJS := $(CXXSRC:.cpp=.o)

# define the executable file 
MAIN := $(BUILDDIR)/cvGrasp

###############
# Build Rules #
###############

.PHONY: depend clean

all:    $(MAIN)
	@echo UAR has been compiled.

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(C_REGULAR_NOT_CPP_BECAUSE_THE_ABOVE_IS_NAMED_CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@for file in $(OBJS) $(MAIN); do if [ -f $$file ]; then rm $$file; fi; done
	@for dir in $(PROJDIRS); do if [ -d $(BUILDDIR)/$$dir ]; then rm -rf $(BUILDDIR)/$$dir; fi; done

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
