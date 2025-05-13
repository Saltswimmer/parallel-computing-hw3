PROGRAM = GameOfLife
SOURCES = main.c
OBJECTS = main.o
CFLAGS = -Wall -O3

.SUFFIXES:
.SUFFIXES: .c .o

.c.o: ; mpicc $(CFLAGS) -c $<

$(PROGRAM) : $(OBJECTS)
	mpicc -o $(PROGRAM) $(CFLAGS) $(OBJECTS)

.PHONY: clean
clean: ; /bin/rm -f $(PROGRAM) $(OBJECTS) output.txt
       
