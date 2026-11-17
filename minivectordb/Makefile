CXX ?= g++
BASE_FLAGS ?= -O3 -std=c++17 -DNDEBUG -Wall -Wextra
# Static linking keeps the runtime image free of shared-library reads: the
# judge charges the container's block I/O and page cache, so fewer libraries
# touched at startup is directly less of both.  The flag is optional so a
# toolchain without static archives still produces working binaries.
STATIC_FLAGS ?= -static -s
AVX_FLAGS ?= -mavx2 -mfma
# 512-bit byte dot products need VNNI; the entrypoint only selects this build
# when /proc/cpuinfo advertises the same features.  The target is optional: a
# toolchain without these flags must not fail the whole image.
AVX512_FLAGS ?= -mavx512f -mavx512bw -mavx512vl -mavx512vnni
SRCS := src/main.cpp src/parser.cpp src/storage.cpp src/engine.cpp src/topk.cpp
HDRS := src/mvdb.h src/distance.h src/string_scan.h

all: generic avx2 avx512

# $(1) = output name, $(2) = extra flags
define build
	@$(CXX) $(BASE_FLAGS) $(STATIC_FLAGS) $(2) -o $(1) $(SRCS) \
	    || $(CXX) $(BASE_FLAGS) $(2) -o $(1) $(SRCS)
endef

generic: $(SRCS) $(HDRS)
	$(call build,minivectordb,)

avx2: $(SRCS) $(HDRS)
	$(call build,minivectordb-avx2,$(AVX_FLAGS))

avx512: $(SRCS) $(HDRS)
	@$(CXX) $(BASE_FLAGS) $(STATIC_FLAGS) $(AVX_FLAGS) $(AVX512_FLAGS) -o minivectordb-avx512 $(SRCS) \
	    || $(CXX) $(BASE_FLAGS) $(AVX_FLAGS) $(AVX512_FLAGS) -o minivectordb-avx512 $(SRCS) \
	    || (echo "note: AVX-512 build skipped (toolchain or flags unavailable)" >&2; \
	        rm -f minivectordb-avx512)

clean:
	rm -f minivectordb minivectordb-avx2 minivectordb-avx512 \
	      minivectordb.exe minivectordb-avx2.exe minivectordb-avx512.exe

.PHONY: all generic avx2 avx512 clean
