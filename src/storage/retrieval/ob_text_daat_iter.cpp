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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"
#include "lib/hash/ob_hashset.h"
#include <new>

namespace oceanbase
{
namespace storage
{
namespace
{
struct FtsBitmapOverflow
{
  common::hash::ObHashSet<uint64_t> set_;
};
}
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (param.base_param_->need_project_relevance()
             && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
    const auto &id_meta = iter_param_->id_proj_expr_->datum_meta_;
    binary_doc_ids_ = id_meta.type_ == common::ObVarcharType
        && id_meta.cs_type_ == common::CS_TYPE_BINARY;
    // A few sorted lists can be merged without a heap or relevance collector.
    small_posting_union_ = mode_flag_ == sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE
        && !function_lookup_mode_ && param.minimum_should_match_ <= 1
        && !iter_param_->need_project_relevance() && !iter_param_->need_filter()
        && iter_param_->relevance_expr_ == nullptr && iter_param_->dim_weights_ == nullptr
        && !iter_param_->limit_param_->is_valid()
        && dim_iters_->count() > 0 && dim_iters_->count() <= 4
        && (binary_doc_ids_ || id_meta.type_ == common::ObUInt64Type);
    skip_doc_id_projection_ = small_posting_union_ && param.skip_doc_id_projection_
        && iter_param_->eval_ctx_->is_vectorized();
    if (small_posting_union_) {
      for (int64_t i = 0; i < dim_iters_->count(); ++i) {
        iter_domain_ids_[i] = nullptr;
      }
    }
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  reset_batches();
  reset_bitmap();
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  reset_batches();
  reset_bitmap();
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
  if (small_posting_union_) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      iter_domain_ids_[i] = nullptr;
    }
  }
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (iter_param_->need_project_relevance()) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

int ObTextDaaTIter::compare_posting_ids(const ObDatum &left, const ObDatum &right) const
{
  if (!binary_doc_ids_) {
    return left.get_uint() < right.get_uint() ? -1 : (left.get_uint() > right.get_uint());
  }
  // Same byte ordering as the binary VARCHAR comparator, including a length tie-break.
  const int cmp = MEMCMP(left.ptr_, right.ptr_, MIN(left.len_, right.len_));
  return cmp == 0 ? (left.len_ < right.len_ ? -1 : (left.len_ > right.len_)) : cmp;
}

void ObTextDaaTIter::reset_batches()
{
  batch_inited_ = false;
  for (int64_t i = 0; i < MAX_BATCH_DIMS; ++i) {
    batch_ids_[i] = nullptr;
    batch_cnt_[i] = 0;
    batch_pos_[i] = 0;
    batch_done_[i] = false;
  }
}

void ObTextDaaTIter::reset_bitmap()
{
  if (OB_NOT_NULL(overflow_ids_)) {
    static_cast<FtsBitmapOverflow *>(overflow_ids_)->set_.destroy();
    overflow_ids_ = nullptr;
  }
  bitmap_ = nullptr;
  bitmap_words_ = 0;
  bitmap_remaining_ = 0;
  bitmap_ready_ = false;
}

int ObTextDaaTIter::ensure_bitmap(const int64_t min_bits)
{
  int ret = OB_SUCCESS;
  int64_t need_words = (min_bits + 63) / 64;
  if (need_words < BITMAP_INITIAL_WORDS) {
    need_words = BITMAP_INITIAL_WORDS;
  }
  if (need_words > bitmap_words_) {
    if (OB_UNLIKELY(need_words > BITMAP_MAX_WORDS)) {
      ret = OB_SIZE_OVERFLOW;
    } else {
      int64_t new_words = bitmap_words_ == 0 ? BITMAP_INITIAL_WORDS : bitmap_words_;
      while (new_words < need_words && new_words < BITMAP_MAX_WORDS) {
        new_words *= 2;
      }
      if (new_words > BITMAP_MAX_WORDS) {
        new_words = BITMAP_MAX_WORDS;
      }
      uint64_t *buf = static_cast<uint64_t *>(
          iter_allocator_->alloc(new_words * static_cast<int64_t>(sizeof(uint64_t))));
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMSET(buf, 0, new_words * static_cast<int64_t>(sizeof(uint64_t)));
        if (OB_NOT_NULL(bitmap_)) {
          MEMCPY(buf, bitmap_, bitmap_words_ * static_cast<int64_t>(sizeof(uint64_t)));
        }
        bitmap_ = buf;
        bitmap_words_ = new_words;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::set_bitmap_id(const uint64_t id)
{
  int ret = OB_SUCCESS;
  if (id < static_cast<uint64_t>(bitmap_words_) * 64ULL) {
    bitmap_[id >> 6] |= (1ULL << (id & 63ULL));
  } else if (id < static_cast<uint64_t>(BITMAP_MAX_WORDS) * 64ULL) {
    if (OB_FAIL(ensure_bitmap(static_cast<int64_t>(id) + 1))) {
    } else {
      bitmap_[id >> 6] |= (1ULL << (id & 63ULL));
    }
  } else {
    FtsBitmapOverflow *overflow = static_cast<FtsBitmapOverflow *>(overflow_ids_);
    if (OB_ISNULL(overflow)) {
      void *buf = iter_allocator_->alloc(sizeof(FtsBitmapOverflow));
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        overflow = new (buf) FtsBitmapOverflow();
        if (OB_FAIL(overflow->set_.create(1024))) {
        } else {
          overflow_ids_ = overflow;
        }
      }
    }
    if (OB_SUCC(ret)) {
      ret = overflow->set_.set_refactored(id);
      if (OB_HASH_EXIST == ret) {
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::build_bitmap_count(int64_t &total)
{
  int ret = OB_SUCCESS;
  total = 0;
  if (OB_FAIL(ensure_bitmap(0))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
      auto *dim = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
      while (OB_SUCC(ret)) {
        const ObDatum *ids = nullptr;
        int64_t cnt = 0;
        ret = dim->get_next_raw_datum_batch(ids, cnt);
        if (OB_ITER_END == ret) {
          ret = OB_SUCCESS;
          break;
        } else if (OB_FAIL(ret)) {
        } else {
          for (int64_t j = 0; OB_SUCC(ret) && j < cnt; ++j) {
            if (OB_FAIL(set_bitmap_id(ids[j].get_uint()))) {
            }
          }
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    int64_t count = 0;
    for (int64_t i = 0; i < bitmap_words_; ++i) {
      count += __builtin_popcountll(bitmap_[i]);
    }
    if (OB_NOT_NULL(overflow_ids_)) {
      count += static_cast<FtsBitmapOverflow *>(overflow_ids_)->set_.size();
    }
    total = count;
  }
  return ret;
}

int ObTextDaaTIter::fetch_batch(const int64_t idx)
{
  int ret = OB_SUCCESS;
  const ObDatum **ids = nullptr;
  int64_t cnt = 0;
  auto *dim = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(idx));
  if (OB_ISNULL(dim)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(dim->get_next_datum_batch(ids, cnt))) {
    if (OB_ITER_END == ret) {
      batch_ids_[idx] = nullptr;
      batch_cnt_[idx] = 0;
      batch_pos_[idx] = 0;
      batch_done_[idx] = true;
      ret = OB_SUCCESS;
    }
  } else if (OB_UNLIKELY(cnt <= 0 || OB_ISNULL(ids))) {
    batch_ids_[idx] = nullptr;
    batch_cnt_[idx] = 0;
    batch_pos_[idx] = 0;
    batch_done_[idx] = true;
  } else {
    batch_ids_[idx] = ids;
    batch_cnt_[idx] = cnt;
    batch_pos_[idx] = 0;
    batch_done_[idx] = false;
  }
  return ret;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(capacity <= 0)) {
  } else if (0 == dim_iters_->count()) {
    ret = OB_ITER_END;
  } else if (!skip_doc_id_projection_ || !iter_param_->eval_ctx_->is_vectorized()
             || dim_iters_->count() > MAX_BATCH_DIMS) {
    ret = ObSRDaaTIterImpl::get_next_rows(capacity, count);
  } else if (iter_param_->limit_param_->is_valid()
             && output_row_cnt_ >= iter_param_->limit_param_->limit_) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(pre_process())) {
    if (OB_ITER_END == ret) {
      count = 0;
    }
  } else if (!binary_doc_ids_) {
    if (!bitmap_ready_) {
      int64_t total = 0;
      if (OB_FAIL(build_bitmap_count(total))) {
      } else {
        bitmap_remaining_ = total;
        bitmap_ready_ = true;
      }
    }
    if (OB_SUCC(ret)) {
      int64_t produced = MIN(capacity, bitmap_remaining_);
      if (produced > 0) {
        bitmap_remaining_ -= produced;
        count = produced;
        input_row_cnt_ += produced;
        output_row_cnt_ += produced;
      } else {
        ret = OB_ITER_END;
      }
    }
  } else {
    if (!batch_inited_) {
      for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
        if (OB_FAIL(fetch_batch(i))) {
        }
      }
      batch_inited_ = OB_SUCC(ret);
    }
    int64_t produced = 0;
    while (OB_SUCC(ret) && produced < capacity) {
      int64_t min_idx = -1;
      for (int64_t i = 0; i < dim_iters_->count(); ++i) {
        if (!batch_done_[i] && batch_pos_[i] < batch_cnt_[i]) {
          if (min_idx < 0
              || compare_posting_ids(*batch_ids_[i][batch_pos_[i]],
                                     *batch_ids_[min_idx][batch_pos_[min_idx]]) < 0) {
            min_idx = i;
          }
        }
      }
      if (min_idx < 0) {
        break;
      }
      // Keep the winner by value: fetching the next batch for its dimension
      // would otherwise invalidate a pointer into the old datum array.
      const ObDatum min_id = *batch_ids_[min_idx][batch_pos_[min_idx]];
      ++produced;
      for (int64_t i = 0; i < dim_iters_->count(); ++i) {
        if (!batch_done_[i]) {
          while (batch_pos_[i] < batch_cnt_[i]
                 && 0 == compare_posting_ids(*batch_ids_[i][batch_pos_[i]], min_id)) {
            ++batch_pos_[i];
          }
        }
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
        if (!batch_done_[i] && batch_pos_[i] >= batch_cnt_[i]) {
          if (OB_FAIL(fetch_batch(i))) {
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (produced > 0) {
        count = produced;
        input_row_cnt_ += produced;
        output_row_cnt_ += produced;
      } else {
        ret = OB_ITER_END;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::do_one_merge_round(int64_t &count)
{
  if (!small_posting_union_) {
    return ObSRDaaTIterImpl::do_one_merge_round(count);
  }

  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
    const int64_t idx = next_round_iter_idxes_[i];
    auto *dim = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(idx));
    if (OB_FAIL(dim->get_next_row())) {
      if (ret == OB_ITER_END) {
        iter_domain_ids_[idx] = nullptr;
        ret = OB_SUCCESS;
      }
    } else if (OB_FAIL(dim->get_curr_id(iter_domain_ids_[idx]))) {
    } else if (OB_ISNULL(iter_domain_ids_[idx]) || iter_domain_ids_[idx]->is_null()) {
      ret = OB_ERR_UNEXPECTED;
    }
  }

  const ObDatum *min_id = nullptr;
  next_round_cnt_ = 0;
  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    const ObDatum *id = iter_domain_ids_[i];
    if (id != nullptr) {
      const int cmp = min_id == nullptr ? -1 : compare_posting_ids(*id, *min_id);
      if (cmp < 0) {
        min_id = id;
        next_round_cnt_ = 0;
      }
      if (cmp <= 0) {
        next_round_iter_idxes_[next_round_cnt_++] = i;
      }
    }
  }

  if (OB_FAIL(ret)) {
  } else if (min_id == nullptr) {
    ret = OB_ITER_END;
  } else if (skip_doc_id_projection_) {
    ++count;
    ++input_row_cnt_;
    ++output_row_cnt_;
  } else if (OB_FAIL(cache_result(count, *min_id, 0.0))) {
  }
  return ret;
}

int ObTextDaaTIter::project_results(int64_t count)
{
  return skip_doc_id_projection_ ? OB_SUCCESS : ObSRDaaTIterImpl::project_results(count);
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
