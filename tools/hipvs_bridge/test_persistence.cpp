#include "seekdb_cuvs_bridge.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static int check_graph_roundtrip()
{
  constexpr long count = 10000;
  constexpr long dim = 128;
  constexpr long nq = 16;
  constexpr long k = 100;
  std::mt19937 random(17);
  std::uniform_real_distribution<float> sample(-1, 1);
  std::vector<float> vectors(count * dim);
  for (float &value : vectors) value = sample(random);
  const std::vector<float> queries(vectors.begin(), vectors.begin() + nq * dim);
  using Index = std::unique_ptr<void, decltype(&seekdb_cuvs_free)>;
  Index original(seekdb_cuvs_build(vectors.data(), count, dim), &seekdb_cuvs_free);
  if (!original) return 3;
  std::array<unsigned int, nq * k> before{}, after{};
  std::array<float, nq * k> before_dist{}, after_dist{};
  if (seekdb_cuvs_search(original.get(), queries.data(), nq, k, before.data(), before_dist.data()) != 0) return 4;
  int32_t degree = 0;
  std::unique_ptr<uint32_t, decltype(&std::free)> graph(
      seekdb_cuvs_export_graph(original.get(), &degree), &std::free);
  if (!graph || degree <= 0 || degree >= count) return 5;
  original.reset();
  Index restored(seekdb_cuvs_restore_graph(vectors.data(), graph.get(), count, dim, degree),
                 &seekdb_cuvs_free);
  graph.reset();
  std::vector<float>().swap(vectors);
  if (!restored ||
      seekdb_cuvs_search(restored.get(), queries.data(), nq, k, after.data(), after_dist.data()) != 0) return 6;
  if (before != after) return 7;
  for (size_t i = 0; i < before_dist.size(); ++i) {
    if (!std::isfinite(after_dist[i]) || std::fabs(before_dist[i] - after_dist[i]) > 0.0001f) return 8;
  }
  std::printf("graph roundtrip: n=%ld dim=%ld degree=%d verified_top100_queries=%ld\n",
              count, dim, degree, nq);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc == 2 && std::strcmp(argv[1], "graph") == 0) return check_graph_roundtrip();
  if (argc != 3) return 2;
  const long count = 10000;
  const long dimension = 32;
  std::vector<float> vectors(count * dimension);
  for (long row = 0; row < count; ++row) {
    for (long column = 0; column < dimension; ++column) {
      vectors[row * dimension + column] = std::sin(float(row * 31 + column * 17));
    }
  }
  void *index = nullptr;
  if (std::strcmp(argv[1], "write") == 0) {
    index = seekdb_cuvs_build(vectors.data(), count, dimension);
    if (!index || seekdb_cuvs_serialize(index, argv[2]) != 0) return 3;
  } else {
    void *invalid = seekdb_cuvs_deserialize(argv[2], count - 1, dimension);
    if (invalid) { seekdb_cuvs_free(invalid); return 6; }
    invalid = seekdb_cuvs_deserialize(argv[2], count, dimension + 1);
    if (invalid) { seekdb_cuvs_free(invalid); return 7; }
    index = seekdb_cuvs_deserialize(argv[2], count, dimension);
    if (!index) return 4;
  }
  unsigned int ids[10];
  float distances[10];
  int result = seekdb_cuvs_search(index, vectors.data(), 1, 10, ids, distances);
  seekdb_cuvs_free(index);
  if (result != 0 || ids[0] != 0 || std::fabs(distances[0]) > 0.0001f) return 5;
  std::printf("%s: n=%ld top1=%u distance=%g\n", argv[1], count, ids[0], distances[0]);
  return 0;
}
