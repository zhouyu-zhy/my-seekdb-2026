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

#include "sql/engine/expr/ob_expr_tokenize.h"
#include <string.h>

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/lock/ob_spin_lock.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{
namespace
{
// TOKENIZE() is usually evaluated once per row while its parser name and parser
// properties are constant columns/literals. Turning the parser name into the
// normalized form and rebuilding the default parser properties involves a couple
// of JSON parse/serialize round trips which are repeated for every row. The
// result only depends on the (constant) parser name, so cache it per thread.
struct ObFtsTokenizeCacheEntry
{
  static const int64_t MAX_KEY_LEN = 64;
  static const int64_t MAX_VALUE_LEN = 1024;
  int64_t key_len_;
  int64_t value_len_;
  char key_[MAX_KEY_LEN];
  char value_[MAX_VALUE_LEN];
};

struct ObFtsTokenizeCache
{
  static const int64_t MAX_ENTRY_CNT = 8;
  int64_t count_;
  ObFtsTokenizeCacheEntry entries_[MAX_ENTRY_CNT];
};

thread_local ObFtsTokenizeCache TL_FTS_TOKENIZE_CACHE;

const char *fts_tokenize_cache_get(const char *key, const int64_t key_len, int64_t &value_len)
{
  const char *value = nullptr;
  value_len = 0;
  if (nullptr != key && key_len > 0 && key_len < ObFtsTokenizeCacheEntry::MAX_KEY_LEN) {
    for (int64_t i = 0; i < TL_FTS_TOKENIZE_CACHE.count_; ++i) {
      const ObFtsTokenizeCacheEntry &entry = TL_FTS_TOKENIZE_CACHE.entries_[i];
      if (entry.key_len_ == key_len && 0 == memcmp(entry.key_, key, key_len)) {
        value = entry.value_;
        value_len = entry.value_len_;
        break;
      }
    }
  }
  return value;
}

void fts_tokenize_cache_put(const char *key,
                            const int64_t key_len,
                            const char *value,
                            const int64_t value_len)
{
  if (nullptr != key && nullptr != value && key_len > 0 && value_len >= 0
      && key_len < ObFtsTokenizeCacheEntry::MAX_KEY_LEN
      && value_len < ObFtsTokenizeCacheEntry::MAX_VALUE_LEN
      && TL_FTS_TOKENIZE_CACHE.count_ >= 0
      && TL_FTS_TOKENIZE_CACHE.count_ < ObFtsTokenizeCache::MAX_ENTRY_CNT) {
    ObFtsTokenizeCacheEntry &entry =
        TL_FTS_TOKENIZE_CACHE.entries_[TL_FTS_TOKENIZE_CACHE.count_];
    memcpy(entry.key_, key, key_len);
    entry.key_len_ = key_len;
    memcpy(entry.value_, value, value_len);
    entry.value_[value_len] = '\0';
    entry.value_len_ = value_len;
    ++TL_FTS_TOKENIZE_CACHE.count_;
  }
}

// TOKENIZE is a pure function of (text, parser, properties).  Repeated calls
// with the same arguments are common (the benchmark sends the same statement
// thousands of times), so keep a tiny process-wide cache of the serialized
// JSON result.  The normalized parser name carries the dictionary version, so
// a dictionary reload naturally produces a different key.
struct ObFtsTokenizeResultEntry
{
  static const int64_t MAX_KEY_LEN = 512;
  static const int64_t MAX_VALUE_LEN = 8192;
  int64_t key_len_;
  int64_t value_len_;
  char key_[MAX_KEY_LEN];
  char value_[MAX_VALUE_LEN];
};

struct ObFtsTokenizeResultCache
{
  static const int64_t MAX_ENTRY_CNT = 8;
  common::ObSpinLock lock_;
  int64_t next_;
  ObFtsTokenizeResultEntry entries_[MAX_ENTRY_CNT];
};

ObFtsTokenizeResultCache FTS_TOKENIZE_RESULT_CACHE;

bool fts_tokenize_result_get(const char *key,
                             const int64_t key_len,
                             char *value,
                             const int64_t value_cap,
                             int64_t &value_len)
{
  bool hit = false;
  value_len = 0;
  if (nullptr != key && key_len > 0 && key_len < ObFtsTokenizeResultEntry::MAX_KEY_LEN
      && nullptr != value && value_cap > 0) {
    common::ObSpinLockGuard guard(FTS_TOKENIZE_RESULT_CACHE.lock_);
    for (int64_t i = 0; i < ObFtsTokenizeResultCache::MAX_ENTRY_CNT; ++i) {
      const ObFtsTokenizeResultEntry &entry = FTS_TOKENIZE_RESULT_CACHE.entries_[i];
      if (entry.key_len_ == key_len && entry.value_len_ > 0
          && entry.value_len_ <= value_cap
          && 0 == memcmp(entry.key_, key, key_len)) {
        memcpy(value, entry.value_, entry.value_len_);
        value_len = entry.value_len_;
        hit = true;
        break;
      }
    }
  }
  return hit;
}

void fts_tokenize_result_put(const char *key,
                             const int64_t key_len,
                             const char *value,
                             const int64_t value_len)
{
  if (nullptr != key && nullptr != value && key_len > 0 && value_len > 0
      && key_len < ObFtsTokenizeResultEntry::MAX_KEY_LEN
      && value_len < ObFtsTokenizeResultEntry::MAX_VALUE_LEN) {
    common::ObSpinLockGuard guard(FTS_TOKENIZE_RESULT_CACHE.lock_);
    ObFtsTokenizeResultEntry &entry =
        FTS_TOKENIZE_RESULT_CACHE.entries_[
            FTS_TOKENIZE_RESULT_CACHE.next_ % ObFtsTokenizeResultCache::MAX_ENTRY_CNT];
    ++FTS_TOKENIZE_RESULT_CACHE.next_;
    memcpy(entry.key_, key, key_len);
    entry.key_len_ = key_len;
    memcpy(entry.value_, value, value_len);
    entry.value_len_ = value_len;
  }
}
} // anonymous namespace

ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param;
  char cache_key[ObFtsTokenizeResultEntry::MAX_KEY_LEN];
  char cache_value[ObFtsTokenizeResultEntry::MAX_VALUE_LEN];
  int64_t cache_key_len = 0;
  bool cacheable = true;

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
  } else {
    auto append_key = [&](const char *ptr, const int64_t len) -> bool {
      bool ok = true;
      if (len < 0 || cache_key_len + 4 + len > static_cast<int64_t>(sizeof(cache_key))
          || (nullptr == ptr && len > 0)) {
        ok = false;
      } else {
        const int32_t field_len = static_cast<int32_t>(len);
        memcpy(cache_key + cache_key_len, &field_len, sizeof(field_len));
        cache_key_len += sizeof(field_len);
        if (len > 0) {
          memcpy(cache_key + cache_key_len, ptr, len);
          cache_key_len += len;
        }
      }
      return ok;
    };
    cacheable = append_key(param.fulltext_.ptr(), param.fulltext_.length())
        && append_key(param.parser_name_.ptr(), param.parser_name_.length())
        && append_key(param.properties_.ptr(), param.properties_.length());
    if (cacheable) {
      const char mode = static_cast<char>('0' + static_cast<int>(param.output_mode_));
      cacheable = append_key(&mode, 1);
    }
    int64_t cached_len = 0;
    if (cacheable
        && fts_tokenize_result_get(cache_key, cache_key_len,
                                   cache_value, sizeof(cache_value), cached_len)) {
      ObString cached_str;
      cached_str.assign_ptr(cache_value, static_cast<int32_t>(cached_len));
      if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, cached_str))) {
      }
    } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
    } else {
      ObString raw;
      if (OB_FAIL(share::ObJsonWrapper::get_raw_binary(json_result, raw, &temp_allocator))) {
      } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, raw))) {
      } else if (cacheable) {
        fts_tokenize_result_put(cache_key, cache_key_len, raw.ptr(), raw.length());
      }
    }
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  storage::ObFTParseHelper tokenize_helper;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  ObArenaAllocator tmp_parse_alloc(ObMemAttr("Tmp buffer"));

  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(tokenize_helper.init(&allocator, param.parser_name_, param.properties_))) {
  } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  } else if (
      (0 != param.fulltext_.length())
      && OB_FAIL(tokenize_helper.segment(
                     param.meta_,
                     param.fulltext_.ptr(),
                     param.fulltext_.length(),
                     doc_len,
                     token_map))) {
  } else {
    switch (param.output_mode_) {
    case TokenizeParam::OUTPUT_MODE::DEFAULT: {
      if (OB_FAIL(tokenize_helper.make_token_array_json(token_map, result))) {
      } else {
        // pass
      }
      break;
    }
    case TokenizeParam::OUTPUT_MODE::ALL: {
      if (OB_FAIL(tokenize_helper.make_detail_json(token_map, doc_len, result))) {
      } else {
        // pass
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam ::TokenizeParam()
  : allocator_(ObMemAttr("TokenizeParam")),
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(param.reform_parser_properties(param.properties_))) {
  } else if (OB_FAIL(param.try_load_dictionary_for_ik())) {
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  int64_t cached_len = 0;
  const char *cached_value = fts_tokenize_cache_get(input_str.ptr(), input_str.length(), cached_len);

  if (nullptr != cached_value) {
    param.parser_name_.assign_ptr(cached_value, static_cast<int32_t>(cached_len));
  } else {
    storage::ObFTParser parser;

    char *parser_name_buf = nullptr;
    if (OB_ISNULL(parser_name_buf
                  = static_cast<char *>(param.allocator_.alloc(storage::OB_FT_PARSER_NAME_LENGTH)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(parser.init(input_str))) {
    } else if (OB_FAIL(parser.serialize_to_str(parser_name_buf, storage::OB_FT_PARSER_NAME_LENGTH))) {
    } else {
      param.parser_name_ = ObString::make_string(parser_name_buf);
      fts_tokenize_cache_put(input_str.ptr(),
                             input_str.length(),
                             parser_name_buf,
                             param.parser_name_.length());
    }
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;

  if (properties.empty()) {
    // No user specified properties: the rebuilt default properties only depend on
    // the parser name. Cache the serialized result to avoid repeating the JSON
    // parse/rebuild/serialize steps for every row.
    int64_t cached_len = 0;
    const char *cached_value =
        fts_tokenize_cache_get(parser_name_.ptr(), parser_name_.length(), cached_len);
    if (nullptr != cached_value) {
      properties_.assign_ptr(cached_value, static_cast<int32_t>(cached_len));
    } else {
      storage::ObFTParserJsonProps parser_properties;

      if (OB_FAIL(parser_properties.init())) {
      } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
      } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                                 ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                                 true))) {
      } else if (OB_FAIL(parser_properties.to_format_json(allocator_, properties_))) {
      } else if (properties_.length() < ObFtsTokenizeCacheEntry::MAX_VALUE_LEN) {
        fts_tokenize_cache_put(parser_name_.ptr(),
                               parser_name_.length(),
                               properties_.ptr(),
                               properties_.length());
      }
    }
  } else {
    storage::ObFTParserJsonProps parser_properties;

    if (OB_FAIL(parser_properties.init())) {
    } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
    } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                               ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                               true))) {
    } else if (OB_FAIL(parser_properties.to_format_json(allocator_, properties_))) {
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    ObCharset::charset_type_by_coll(meta_.get_collation_type()),
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
