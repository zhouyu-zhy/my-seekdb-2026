// [competition] ob_ft_dict_hub.cpp
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

#define USING_LOG_PREFIX STORAGE_FTS

#include "storage/fts/dict/ob_ft_dict_hub.h"

#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
namespace oceanbase
{
namespace storage
{
class BuiltinIKDictionary
{
public:
  BuiltinIKDictionary()
    : allocator_(lib::ObMemAttr("IKDictionary")),
      main_ranges_(allocator_),
      quantifier_ranges_(allocator_),
      main_desc_("main_dict", ObFTDictType::DICT_IK_MAIN, CHARSET_UTF8MB4, CS_TYPE_UTF8MB4_BIN),
      quantifier_desc_("quan_dict", ObFTDictType::DICT_IK_QUAN, CHARSET_UTF8MB4, CS_TYPE_UTF8MB4_BIN),
      main_(allocator_, &main_ranges_, main_desc_),
      quantifier_(allocator_, &quantifier_ranges_, quantifier_desc_)
  {}

  int init(ObFTDictHub &hub)
  {
    int ret = OB_SUCCESS;
    if (OB_FAIL(load(hub, main_desc_, main_ranges_))) {
    } else if (OB_FAIL(load(hub, quantifier_desc_, quantifier_ranges_))) {
    } else if (OB_FAIL(main_.init())) {
    } else if (OB_FAIL(quantifier_.init())) {
    }
    return ret;
  }

  const ObIFTDict *main() const { return &main_; }
  const ObIFTDict *quantifier() const { return &quantifier_; }

private:
  static int load(ObFTDictHub &hub, const ObFTDictDesc &desc, ObFTCacheRangeContainer &ranges)
  {
    int ret = hub.load_cache(desc, ranges);
    if (OB_ENTRY_NOT_EXIST == ret) {
      ret = hub.build_cache(desc, ranges);
    }
    return ret;
  }

  ObArenaAllocator allocator_;
  ObFTCacheRangeContainer main_ranges_;
  ObFTCacheRangeContainer quantifier_ranges_;
  ObFTDictDesc main_desc_;
  ObFTDictDesc quantifier_desc_;
  ObFTRangeDict main_;
  ObFTRangeDict quantifier_;
};

int ObFTDictHub::get_builtin_ik(const ObIFTDict *&main_dict, const ObIFTDict *&quantifier_dict)
{
  int ret = OB_SUCCESS;
  BuiltinIKDictionary *dictionary = builtin_ik_.load(std::memory_order_acquire);
  if (nullptr == dictionary) {
    lib::ObMutexGuard guard(builtin_ik_lock_);
    dictionary = builtin_ik_.load(std::memory_order_relaxed);
    if (nullptr == dictionary) {
      if (OB_ISNULL(dictionary = OB_NEW(BuiltinIKDictionary, "IKDictionary"))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(dictionary->init(*this))) {
        OB_DELETE(BuiltinIKDictionary, "IKDictionary", dictionary);
        dictionary = nullptr;
      } else {
        builtin_ik_.store(dictionary, std::memory_order_release);
      }
    }
  }
  if (OB_SUCC(ret)) {
    main_dict = dictionary->main();
    quantifier_dict = dictionary->quantifier();
  }
  return ret;
}

int ObFTDictHub::init()
{
  static constexpr int K_MAX_DICT_BUCKET = 128; // for now, only built-in dicts.
  int ret = OB_SUCCESS;
  if (OB_FAIL(dict_map_.create(K_MAX_DICT_BUCKET, "dict_map"))) {
  } else if (OB_FAIL(rw_dict_lock_.init(K_MAX_DICT_BUCKET))) {
  } else {
    is_inited_ = true;
  }
  return ret;
};

int ObFTDictHub::destroy()
{
  int ret = OB_SUCCESS;
  BuiltinIKDictionary *dictionary = builtin_ik_.exchange(nullptr, std::memory_order_acq_rel);
  if (nullptr != dictionary) {
    OB_DELETE(BuiltinIKDictionary, "IKDictionary", dictionary);
  }
  is_inited_ = false;
  return ret;
}
int ObFTDictHub::build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  ObFTDictInfo info;
  container.reset();

  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    ObBucketHashWLockGuard guard(rw_dict_lock_, key.hash());

    // try if valid with no recursive lock
    if (OB_FAIL(get_dict_info(key, info))) {
      if (OB_HASH_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
        ret = OB_ENTRY_NOT_EXIST;
      } else {
      }
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
      } else {
      }
    }

    if (OB_FAIL(ret)) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        if (OB_FAIL(ObFTRangeDict::build_cache_from_ik_dict(desc, container))) {
        } else if (FALSE_IT(info.range_count_ = container.get_handles().size())) {
        } else if (OB_FAIL(put_dict_info(key, info))) {
        }
      }
    }
  }
  return ret;
}

int ObFTDictHub::load_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfo info;
  container.reset();
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    {
      ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
      if (OB_FAIL(get_dict_info(key, info))) {
        if (OB_HASH_NOT_EXIST == ret) {
          // dict not exist, make new one, by caller
          ret = OB_ENTRY_NOT_EXIST;
        } else {
        }
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
      } else {
      }
    }
  }

  return ret;
}


int ObFTDictHub::get_dict_info(const ObFTDictInfoKey &key, ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;

  if (OB_FAIL(dict_map_.get_refactored(key, info))) {
    if (OB_HASH_NOT_EXIST != ret) {
    }
  }

  return ret;
}

int ObFTDictHub::put_dict_info(const ObFTDictInfoKey &key, const ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;
  const int cover_exist_flag = 1;
  if (OB_FAIL(dict_map_.set_refactored(key, info, cover_exist_flag))) {
  }

  return ret;
}
} //  namespace storage
} //  namespace oceanbase
