// [competition] ob_ft_dict_def.h
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

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_

#include "lib/charset/ob_charset.h"

#include <cstdint>

namespace oceanbase
{
namespace storage
{
typedef int32_t ObFTWordCode;

constexpr uint32_t FTS_CJK_FIRST = 0x4E00;
constexpr uint32_t FTS_CJK_END = 0xA000;
constexpr uint32_t FTS_CJK_COUNT = FTS_CJK_END - FTS_CJK_FIRST;

inline int32_t utf8_cjk_index(const ObString &word)
{
  if (word.length() != 3) {
    return -1;
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(word.ptr());
  if (bytes[0] < 0xE4 || bytes[0] > 0xE9
      || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80) {
    return -1;
  }
  const uint32_t cp = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
  return cp < FTS_CJK_FIRST || cp >= FTS_CJK_END ? -1 : static_cast<int32_t>(cp - FTS_CJK_FIRST);
}
typedef int32_t ObFTWordBase;
typedef uint32_t ObFTWordStateIndex;

/**
 * @class ObFTSingleWord
 * @brief sturct to store a single character of a charset;
 */
struct ObFTSingleWord
{
public:
  ObFTSingleWord() : word(""), word_len(0) {}
  ObFTSingleWord(const ObFTSingleWord &other) = default;
  ObFTSingleWord &operator=(const ObFTSingleWord &other) = default;

  int32_t set_word(const char *word, int32_t word_len);
  ObString get_word() const;
  bool operator==(const ObFTSingleWord &other) const;

public:
  char word[common::ObCharset::MAX_MB_LEN];
  uint8_t word_len;
} __attribute__((packed));

enum class ObFTDictType : uint32_t
{
  DICT_TYPE_INVALID = 0,
  DICT_IK_MAIN = 1,
  DICT_IK_QUAN = 2,
  DICT_IK_STOP = 3,
};

class ObFTDictDesc
{
public:
  ObFTDictDesc(const ObString &name,
               const ObFTDictType type,
               const ObCharsetType charset,
               const ObCollationType coll_type)
      : name_(name), type_(type), charset_(charset), coll_type_(coll_type)
  {
  }

public:
  ObString name_;
  ObFTDictType type_;
  ObCharsetType charset_;
  ObCollationType coll_type_;
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_
