APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_120 -o $@ $<
clean:
	rm -f ${APPS}
