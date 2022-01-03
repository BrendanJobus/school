CC = g++

CFLAGS = -g -Wall

TARGET = ibm_model1_hw_stub

all = $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp
	./$(TARGET)
	$(RM) $(TARGET)

final:
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET)
