// [competition] ob_beng_ft_parser.cpp
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

#include "storage/fts/ob_beng_ft_parser.h"
#include "lib/charset/ob_ctype.h"
#include "data_plane/fts/ob_fts_struct.h"

using namespace oceanbase::common;

namespace oceanbase
{
namespace storage
{
namespace
{
// Unaligned loads through memcpy are safe; the high bit in any byte
// identifies a document that needs the charset-aware analyzer.
bool beng_is_ascii(const char *text, int64_t len)
{
  constexpr uint64_t high_bits = 0x8080808080808080ULL;
  while (len >= static_cast<int64_t>(sizeof(uint64_t))) {
    uint64_t bytes;
    MEMCPY(&bytes, text, sizeof(bytes));
    if ((bytes & high_bits) != 0) {
      return false;
    }
    text += sizeof(bytes);
    len -= sizeof(bytes);
  }
  for (int64_t i = 0; i < len; ++i) {
    if (static_cast<unsigned char>(text[i]) >= 0x80) {
      return false;
    }
  }
  return true;
}
} // namespace

int ObBEngFTParser::get_next_token(
    const char *&word,
    int64_t &word_len,
    int64_t &char_len,
    int64_t &word_freq)
{
  int ret = OB_SUCCESS;
  ObDatum token;
  int64_t token_freq = 0;
  char *buf = nullptr;
  word = nullptr;
  word_len = 0;
  char_len = 0;
  word_freq = 0;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (ascii_cursor_ >= 0) {
    ret = next_ascii_token(word, word_len, char_len, word_freq);
  } else if (OB_ISNULL(token_stream_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(token_stream_->get_next(token, token_freq))) {
    if (OB_ITER_END != ret) {
    }
  } else if (OB_ISNULL(token.ptr_) || OB_UNLIKELY(0 >= token.len_ || 0 >= token_freq)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    if (nullptr != copy_buf_ && copy_pos_ + token.len_ <= copy_cap_) {
      buf = copy_buf_ + copy_pos_;
      copy_pos_ += token.len_;
    } else if (OB_ISNULL(buf = static_cast<char *>(allocator_.alloc(token.len_)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    }
    if (OB_SUCC(ret)) {
      MEMCPY(buf, token.ptr_, token.len_);
      word = buf;
      word_len = token.len_;
      char_len = token.len_;
      word_freq = token_freq;
      LOG_DEBUG("succeed to add word", K(ObString(word_len, word)), K(word_freq));
    }
  }
  return ret;
}

int ObBEngFTParser::next_ascii_token(
    const char *&word, int64_t &word_len, int64_t &char_len, int64_t &word_freq)
{
  constexpr unsigned int delimiter = _MY_SPC | _MY_CTR | _MY_PNT;
  constexpr unsigned int alnum = _MY_U | _MY_L | _MY_NMR;
  const auto *text = reinterpret_cast<const unsigned char *>(doc_.ptr_);
  const auto *ctype = analysis_ctx_.cs_->ctype + 1;
  const int64_t size = doc_.len_;
  while (ascii_cursor_ < size) {
    // Delimiters and edge trimming follow the original tokenizer/normalizer.
    int64_t begin = ascii_cursor_;
    while (ascii_cursor_ < size && !(ctype[text[ascii_cursor_]] & delimiter)) {
      ++ascii_cursor_;
    }
    int64_t end = ascii_cursor_;
    if (ascii_cursor_ < size) {
      ++ascii_cursor_;
    }
    while (begin < end && !(ctype[text[begin]] & alnum)) {
      ++begin;
    }
    while (end > begin && !(ctype[text[end - 1]] & alnum)) {
      --end;
    }
    if (begin < end) {
      char *output = copy_buf_ + copy_pos_;
      for (int64_t i = begin; i < end; ++i) {
        const unsigned char byte = text[i];
        output[i - begin] = static_cast<char>('A' <= byte && byte <= 'Z' ? byte + ('a' - 'A') : byte);
      }
      word = output;
      word_len = end - begin;
      char_len = word_len;
      word_freq = 1;
      copy_pos_ += word_len;
      return OB_SUCCESS;
    }
  }
  return OB_ITER_END;
}

int ObBEngFTParser::init(ObFTParserParam *param)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(is_inited_)) {
    ret = OB_INIT_TWICE;
  } else if (OB_ISNULL(param) || OB_UNLIKELY(!param->is_valid())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_UNLIKELY(UINT32_MAX < param->ft_length_)) {
    ret = OB_NOT_SUPPORTED;
  } else {
    doc_.set_string(param->fulltext_, param->ft_length_);
    copy_cap_ = param->ft_length_ + 16;
    copy_pos_ = 0;
    copy_buf_ = static_cast<char *>(param->allocator_->alloc(copy_cap_));
    if (nullptr == copy_buf_) {
      copy_cap_ = 0;
    }
    analysis_ctx_.cs_ = param->cs_;
    analysis_ctx_.filter_stopword_ = false;
    analysis_ctx_.need_grouping_ = false;
    const bool ascii = copy_buf_ != nullptr && param->cs_->cset == &ob_charset_utf8mb4_handler
        && param->cs_->ctype != nullptr && param->cs_->caseinfo == &ob_unicase_default
        && beng_is_ascii(param->fulltext_, param->ft_length_);
    if (ascii) {
      ascii_cursor_ = 0;
      is_inited_ = true;
    } else if (OB_FAIL(english_analyzer_.init(analysis_ctx_, *param->allocator_))) {
    } else if (OB_FAIL(segment(doc_, token_stream_))) {
    } else if (OB_ISNULL(token_stream_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      is_inited_ = true;
    }
  }
  if (OB_FAIL(ret) && OB_UNLIKELY(!is_inited_)) {
    reset();
  }
  return ret;
}

int ObBEngFTParser::segment(
    const common::ObDatum &doc,
    share::ObITokenStream *&token_stream)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(doc.ptr_) || OB_UNLIKELY(0 >= doc.len_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_UNLIKELY(UINT32_MAX < doc.len_)) {
    ret = OB_NOT_SUPPORTED;
  } else if (OB_FAIL(english_analyzer_.analyze(doc, token_stream))) {
  }
  return ret;
}

void ObBEngFTParser::reset()
{
  analysis_ctx_.reset();
  english_analyzer_.reset();
  doc_.reset();
  token_stream_ = nullptr;
  copy_buf_ = nullptr;
  copy_cap_ = 0;
  copy_pos_ = 0;
  ascii_cursor_ = -1;
  is_inited_ = false;
}

ObBasicEnglishFTParserDesc::ObBasicEnglishFTParserDesc() {}

int ObBasicEnglishFTParserDesc::segment(
    ObFTParserParam *param,
    ObITokenIterator *&iter) const
{
  int ret = OB_SUCCESS;
  ObBEngFTParser *parser = nullptr;
  if (OB_ISNULL(param) || OB_ISNULL(param->fulltext_) || OB_UNLIKELY(!param->is_valid())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(parser = OB_NEWx(ObBEngFTParser, param->allocator_, *(param->allocator_)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(parser->init(param))) {
  } else {
    iter = parser;
  }

  if (OB_FAIL(ret)) {
    OB_DELETEx(ObBEngFTParser, param->allocator_, parser);
  }

  return ret;
}

void ObBasicEnglishFTParserDesc::free_token_iter(
    ObFTParserParam *param,
    ObITokenIterator *&iter) const
{
  if (OB_NOT_NULL(iter)) {
    abort_unless(nullptr != param);
    abort_unless(nullptr != param->allocator_);
    iter->~ObITokenIterator();
    param->allocator_->free(iter);
  }
}

int ObBasicEnglishFTParserDesc::get_add_word_flag(ObAddWordFlag &flag) const
{
  int ret = OB_SUCCESS;
  flag.set_min_max_word();
  flag.set_stop_word();
  flag.set_groupby_word();
  return ret;
}

} // end namespace storage
} // end namespace oceanbase
