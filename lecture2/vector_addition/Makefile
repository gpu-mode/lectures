
NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = vector_addition
OBJ	        = vector_addition.o

default: $(EXE)

vector_addition.o: vector_addition.cu
	$(NVCC) -c -o $@ vector_addition.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
