/*
 * Copyright (c) 2026 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hip/hip_runtime.h>

namespace {
// A wave compacts one row at a time. Ballot preserves IVF's hit order, so the
// adjacency is identical to the host loop even for ties and missing hits.
__global__ void pack_graph(const long long *neighbors, unsigned *graph,
                           long rows, long start, long n, int degree)
{
  const unsigned lane = threadIdx.x % warpSize;
  const unsigned waves = blockDim.x / warpSize;
  const unsigned long long preceding = (1ULL << lane) - 1;
  const int k = degree + 1;
  for (long row = blockIdx.x * waves + threadIdx.x / warpSize;
       row < rows; row += gridDim.x * waves) {
    const long self = start + row;
    int written = 0;
    for (int col = 0; col < k && written < degree; col += warpSize) {
      const int j = col + lane;
      const long long id = j < k ? neighbors[row * k + j] : -1;
      const bool valid = id >= 0 && id != self && id < n;
      const auto mask = __ballot(valid);
      const int offset = written + __builtin_popcountll(mask & preceding);
      if (valid && offset < degree) {
        graph[row * degree + offset] = static_cast<unsigned>(id);
      }
      written += __builtin_popcountll(mask);
    }
    for (int j = written + lane; j < degree; j += warpSize) {
      graph[row * degree + j] = static_cast<unsigned>((self + j + 1) % n);
    }
  }
}
} // namespace

extern "C" int seekdb_cuvs_pack_graph(const long long *neighbors, unsigned *graph,
                                    long rows, long start, long n, int degree,
                                    hipStream_t stream)
{
  constexpr unsigned threads = 128;
  const unsigned blocks = (rows + 3) / 4;
  hipLaunchKernelGGL(pack_graph, dim3(blocks), dim3(threads), 0, stream,
                     neighbors, graph, rows, start, n, degree);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}
