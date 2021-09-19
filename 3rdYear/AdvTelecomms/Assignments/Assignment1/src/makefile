IDIR = ../include
CC = clang++
CFLAGS = -I$(IDIR)

ODIR = obj
LDIR = ../lib

LIBS = -pthread

_DEPS = networking.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = server.o networking.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

server: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~