CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -ffast-math -fopenmp -std=c++17 -pthread
TARGET    = bench
SRC       = bench.cpp
HDRS      = index.hpp streamvec.hpp baselines.hpp

$(TARGET): $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
