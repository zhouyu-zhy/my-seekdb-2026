CXX ?= g++
BASE_FLAGS ?= -O3 -std=c++17 -DNDEBUG -Wall -Wextra
AVX_FLAGS ?= -mavx2 -mfma
# 512-bit byte dot products need VNNI; the entrypoint only selects this build
# when /proc/cpuinfo advertises the same features.  The target is optional: a
# toolchain without these flags must not fail the whole image.
AVX512_FLAGS ?= -mavx512f -mavx512bw -mavx512vl -mavx512vnni
SRCS := src/main.cpp src/parser.cpp src/storage.cpp src/engine.cpp src/topk.cpp
HDRS := src/mvdb.h src/distance.h src/string_scan.h

all: generic avx2 avx512

generic: $(SRCS) $(HDRS)
	$(CXX) $(BASE_FLAGS) -o minivectordb $(SRCS)

avx2: $(SRCS) $(HDRS)
	$(CXX) $(BASE_FLAGS) $(AVX_FLAGS) -o minivectordb-avx2 $(SRCS)

avx512: $(SRCS) $(HDRS)
	@$(CXX) $(BASE_FLAGS) $(AVX_FLAGS) $(AVX512_FLAGS) -o minivectordb-avx512 $(SRCS) \
	    || (echo "note: AVX-512 build skipped (toolchain or flags unavailable)" >&2; \
	        rm -f minivectordb-avx512)

clean:
	rm -f minivectordb minivectordb-avx2 minivectordb-avx512 \
	      minivectordb.exe minivectordb-avx2.exe minivectordb-avx512.exe

.PHONY: all generic avx2 avx512 clean
