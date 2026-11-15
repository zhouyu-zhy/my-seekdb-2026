/*
 * Copyright (c) 2025 OceanBase.
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

#ifndef OB_TEXT_DAAT_ITER_H_
#define OB_TEXT_DAAT_ITER_H_

#include "query/ob_sql_define.h"
#include "ob_inv_idx_param_estimator.h"
#include "ob_sparse_bmw_iter.h"
#include "ob_sparse_daat_iter.h"
#include "ob_text_retrieval_token_iter.h"

namespace oceanbase
{
namespace sql
{
  class ObDASScanIter;
}
namespace storage
{

struct ObTextDaaTParam
{
  ObTextDaaTParam()
    : dim_iters_(nullptr), base_param_(nullptr), allocator_(nullptr), relevance_collector_(nullptr),
      bm25_param_est_ctx_(), mode_flag_(sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE),
      function_lookup_mode_(false), minimum_should_match_(0), skip_doc_id_projection_(false) {}
  ~ObTextDaaTParam() {}
  TO_STRING_KV(K_(base_param), KP_(dim_iters), K_(bm25_param_est_ctx),  K_(mode_flag), K_(function_lookup_mode));
  ObIArray<ObISRDaaTDimIter *> *dim_iters_;
  ObSparseRetrievalMergeParam *base_param_;
  common::ObArenaAllocator *allocator_;
  ObSRDaaTRelevanceCollector *relevance_collector_;
  ObBM25ParamEstCtx bm25_param_est_ctx_;
  sql::ObMatchAgainstMode mode_flag_;
  bool function_lookup_mode_;
  int64_t minimum_should_match_;
  bool skip_doc_id_projection_;
};

class ObTextDaaTIter final : public ObSRDaaTIterImpl
{
public:
  ObTextDaaTIter() : ObSRDaaTIterImpl(),
      bm25_param_estimator_(),
      mode_flag_(sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE),
      function_lookup_mode_(false), small_posting_union_(false), binary_doc_ids_(false),
      skip_doc_id_projection_(false) {}
  virtual ~ObTextDaaTIter() { reset(); }

  int init(const ObTextDaaTParam &param);
  virtual void reuse(const bool switch_tablet = false) override;
  virtual void reset() override;
  virtual int get_next_rows(const int64_t capacity, int64_t &count) override;
protected:
  virtual int pre_process() override;
  virtual int do_one_merge_round(int64_t &count) override;
  virtual int project_results(int64_t count) override;
protected:
  ObBM25ParamEstimator bm25_param_estimator_;
  sql::ObMatchAgainstMode mode_flag_;
  bool function_lookup_mode_;
  bool small_posting_union_;
  bool binary_doc_ids_;
  bool skip_doc_id_projection_;
private:
  int compare_posting_ids(const ObDatum &left, const ObDatum &right) const;
  int fetch_batch(const int64_t idx);
  void reset_batches();
  int ensure_bitmap(const int64_t min_bits);
  int set_bitmap_id(const uint64_t id);
  int build_bitmap_count(int64_t &total);
  void reset_bitmap();
  int add_bitmap_datum(const ObDatum &datum);
  int add_overflow_doc_id(const ObDatum &datum);
  void reset_binary_bitmap();
  static const int64_t MAX_BATCH_DIMS = 4;
  static const int64_t BITMAP_INITIAL_WORDS = 1024;
  static const int64_t BITMAP_MAX_WORDS = 1 << 18;
  const ObDatum **batch_ids_[MAX_BATCH_DIMS] = {};
  int64_t batch_cnt_[MAX_BATCH_DIMS] = {};
  int64_t batch_pos_[MAX_BATCH_DIMS] = {};
  bool batch_done_[MAX_BATCH_DIMS] = {};
  bool batch_inited_ = false;
  uint64_t *bitmap_ = nullptr;
  int64_t bitmap_words_ = 0;
  int64_t bitmap_remaining_ = 0;
  bool bitmap_ready_ = false;
  uint64_t binary_prefix_ = 0;
  bool binary_prefix_set_ = false;
  void *overflow_ids_ = nullptr;
  void *doc_id_overflow_ = nullptr;
  DISALLOW_COPY_AND_ASSIGN(ObTextDaaTIter);
};


class ObTextBMWIter final : public ObSRBMWIterImpl
{
public:
  ObTextBMWIter();
  virtual ~ObTextBMWIter() { reset(); }
  virtual void reuse(const bool switch_tablet = false) override;
  virtual void reset() override;
  int init(const ObTextDaaTParam &param);
protected:
  virtual int get_next_rows(const int64_t capacity, int64_t &count) override;
  virtual int init_before_wand_process() override;
protected:
  ObBM25ParamEstimator bm25_param_estimator_;
private:
  DISALLOW_COPY_AND_ASSIGN(ObTextBMWIter);
};

} // namespace storage
} // namespace oceanbase

#endif
