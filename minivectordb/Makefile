CXX ?= g++
BASE_FLAGS ?= -O3 -std=c++17 -DNDEBUG -Wall -Wextra
AVX_FLAGS ?= -mavx2 -mfma
SRCS := src/main.cpp src/parser.cpp src/storage.cpp src/engine.cpp

all: generic avx2

generic: $(SRCS) src/mvdb.h
	$(CXX) $(BASE_FLAGS) -o minivectordb $(SRCS)

avx2: $(SRCS) src/mvdb.h
	$(CXX) $(BASE_FLAGS) $(AVX_FLAGS) -o minivectordb-avx2 $(SRCS)

clean:
	rm -f minivectordb minivectordb-avx2 minivectordb.exe minivectordb-avx2.exe

.PHONY: all generic avx2 clean
