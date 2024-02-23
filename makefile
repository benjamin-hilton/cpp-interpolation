CC=g++

all: interpolation.cpp utils.h matrix_LU.h
	$(CC) interpolation.cpp -o interpolation.o

clean:
	rm -f interpolation.o
