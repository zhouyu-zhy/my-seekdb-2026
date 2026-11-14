/**
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

#ifndef OCEANBASE_SHARE_SCHEMA_OB_AI_COLUMN_COMMON_H_
#define OCEANBASE_SHARE_SCHEMA_OB_AI_COLUMN_COMMON_H_

#include "lib/string/ob_sql_string.h"
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_table_schema.h"
#include "share/schema/ob_schema_struct.h"

namespace oceanbase
{
namespace share
{

// Row status kept in the hidden __ai_status_<colid> column.
enum ObAiColumnStatus : int64_t
{
  OB_AI_COLUMN_STATUS_PENDING = 0,
  OB_AI_COLUMN_STATUS_READY = 1,
  OB_AI_COLUMN_STATUS_FAILED = 2,
};

// Names of the hidden bookkeeping columns an AI table carries alongside its
// AI columns: the per-row status, the shared invalidation counter and the
// per-row result description.  They are created by the DDL paths and are only
// ever addressed from SQL that the kernel generates itself.
const char *const OB_AI_STATUS_COLUMN_PREFIX = "__ai_status_";
const char *const OB_AI_GEN_COLUMN_NAME = "__ai_gen";
const char *const OB_AI_SPEC_COLUMN_PREFIX = "__ai_spec_";

// True for the three name shapes above.  Used by the column resolver to let
// kernel-generated AI statements refer to those columns while ordinary user
// statements keep getting "unknown column" for them.
inline bool is_ai_column_bookkeeping_name(const common::ObString &name)
{
  return name.prefix_match(OB_AI_STATUS_COLUMN_PREFIX)
      || name.prefix_match(OB_AI_GEN_COLUMN_NAME)
      || name.prefix_match(OB_AI_SPEC_COLUMN_PREFIX);
}

// Task state kept in __all_ai_column_task.state.
enum ObAiColumnTaskState : int64_t
{
  OB_AI_COLUMN_TASK_PENDING = 0,
  OB_AI_COLUMN_TASK_CLAIMED = 2,
  OB_AI_COLUMN_TASK_RETRY_WAIT = 3,
};

// System prompts chosen from the result column type.  The prompt text is also
// persisted into __all_ai_column.normalized_expr so external tools can audit
// exactly what instruction a column was generated with.
const char *const OB_AI_COLUMN_JSON_SYSTEM_PROMPT =
    "Return exactly one valid JSON value.";
const char *const OB_AI_COLUMN_INTEGER_SYSTEM_PROMPT =
    "Return a single base-10 integer.";
const char *const OB_AI_COLUMN_TEXT_SYSTEM_PROMPT =
    "Answer the user request.";

// Returns nullptr for embedding columns (the vector payload needs no prompt).
inline const char *get_ai_column_system_prompt(const schema::ObColumnSchemaV2 &column)
{
  const char *prompt = OB_AI_COLUMN_TEXT_SYSTEM_PROMPT;
  if (column.is_ai_column() && !column.get_ai_column_embed_model().empty()) {
    prompt = nullptr;
  } else {
    const ObObjType type = column.get_data_type();
    if (column.is_json()) {
      prompt = OB_AI_COLUMN_JSON_SYSTEM_PROMPT;
    } else if (ob_is_integer_type(type)
               && ObTinyIntType != type /* tinyint(1) is treated as bool-ish text */) {
      prompt = OB_AI_COLUMN_INTEGER_SYSTEM_PROMPT;
    }
  }
  return prompt;
}

// normalized_expr = user expression text + the derived type prompt, which is
// what the background worker actually sends to the model endpoint.
inline int build_ai_column_normalized_expr(
    const schema::ObColumnSchemaV2 &column,
    common::ObSqlString &out)
{
  int ret = OB_SUCCESS;
  const ObString expr = column.get_ai_column_expr();
  if (OB_FAIL(out.assign(expr))) {
    LOG_WARN("failed to assign ai column expr", K(ret));
  } else {
    const char *prompt = get_ai_column_system_prompt(column);
    if (OB_NOT_NULL(prompt) && OB_FAIL(out.append_fmt(" %s", prompt))) {
      LOG_WARN("failed to append type prompt", K(ret));
    }
  }
  return ret;
}

inline int append_ai_column_sql_escaped(
    const common::ObString &raw,
    common::ObSqlString &out)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < raw.length(); ++i) {
    const char c = raw.ptr()[i];
    if ('\'' == c || '\\' == c) {
      char esc[2] = {'\\', c};
      if (OB_FAIL(out.append(esc, 2))) {
        LOG_WARN("failed to escape char", K(ret));
      }
    } else if ('\n' == c) {
      if (OB_FAIL(out.append("\\n", 2))) {
        LOG_WARN("failed to escape newline", K(ret));
      }
    } else if ('\r' == c) {
      if (OB_FAIL(out.append("\\r", 2))) {
        LOG_WARN("failed to escape carriage return", K(ret));
      }
    } else if ('\0' == c) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("NUL byte in ai column value is not supported", K(ret));
    } else if (OB_FAIL(out.append(&c, 1))) {
      LOG_WARN("failed to append char", K(ret));
    }
  }
  return ret;
}

// Quoted SQL literal for a task rowkey_predicate or an update value.
inline int build_ai_column_sql_literal(
    const common::ObString &raw,
    common::ObSqlString &out)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(out.append("'", 1))) {
  } else if (OB_FAIL(append_ai_column_sql_escaped(raw, out))) {
  } else if (OB_FAIL(out.append("'", 1))) {
  }
  return ret;
}

// Builds the SQL expression that renders a row's primary key as the canonical
// `pk` = value string stored in __all_ai_column_task.rowkey_predicate.
inline int build_ai_column_pk_expr(
    const schema::ObTableSchema &table_schema,
    common::ObSqlString &out)
{
  int ret = OB_SUCCESS;
  ObSEArray<const schema::ObColumnSchemaV2 *, OB_MAX_ROWKEY_COLUMN_NUMBER> rowkeys;
  for (schema::ObTableSchema::const_column_iterator iter = table_schema.column_begin();
       OB_SUCC(ret) && iter != table_schema.column_end(); ++iter) {
    const schema::ObColumnSchemaV2 *column = *iter;
    if (OB_ISNULL(column)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (column->is_rowkey_column()) {
      if (OB_FAIL(rowkeys.push_back(column))) {
        LOG_WARN("failed to push rowkey column", K(ret));
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (rowkeys.empty()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("table without rowkey columns cannot host ai columns", K(ret));
  } else {
    struct ColExpr { ObString name_; bool numeric_; };
    ObSEArray<ColExpr, OB_MAX_ROWKEY_COLUMN_NUMBER> exprs;
    for (int64_t i = 0; OB_SUCC(ret) && i < rowkeys.count(); ++i) {
      const schema::ObColumnSchemaV2 *column = rowkeys.at(i);
      ColExpr item;
      item.name_ = column->get_column_name_str();
      // every pk flavor goes through the quoting/escaping REPLACE chain; the
      // implicit cast renders numbers exactly like CONCAT would.
      item.numeric_ = false;
      if (OB_FAIL(exprs.push_back(item))) {
        LOG_WARN("failed to push pk expr", K(ret));
      }
    }
    if (OB_FAIL(ret)) {
    } else {
      // One outer CONCAT with a trailing empty literal avoids dangling
      // separators: CONCAT('`a` = ', `a`, ' AND ', '`b` = ', <b>, '')
      if (OB_FAIL(out.append("CONCAT(", 7))) {
        LOG_WARN("failed to open pk expr", K(ret));
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < exprs.count(); ++i) {
        const ObString &name = exprs.at(i).name_;
        if (i > 0 && OB_FAIL(out.append("' AND ', ", 9))) {
          LOG_WARN("failed to append pk separator", K(ret));
        } else if (OB_FAIL(out.append("'", 1))
                   || OB_FAIL(out.append("`", 1))
                   || OB_FAIL(out.append(name.ptr(), name.length()))
                   || OB_FAIL(out.append("` = ", 4))
                   || OB_FAIL(out.append("'", 1))
                   || OB_FAIL(out.append(", ", 2))) {
          LOG_WARN("failed to append pk literal part", K(ret));
        } else if (exprs.at(i).numeric_) {
          if (OB_FAIL(out.append("`", 1))
              || OB_FAIL(out.append(name.ptr(), name.length()))
              || OB_FAIL(out.append("`", 1))) {
            LOG_WARN("failed to append pk column ref", K(ret));
          }
        } else if (OB_FAIL(out.append("REPLACE(REPLACE(`", 17))
                   || OB_FAIL(out.append(name.ptr(), name.length()))
                   || OB_FAIL(out.append("`, '\\\\', '\\\\\\\\'), '''', '''''')", 31))) {
          LOG_WARN("failed to append pk string escape", K(ret));
        }
        if (OB_SUCC(ret) && i + 1 < exprs.count() && OB_FAIL(out.append(", ", 2))) {
          LOG_WARN("failed to append arg separator", K(ret));
        }
      }
      if (OB_SUCC(ret)) {
        if (OB_FAIL(out.append(", '')", 6))) {
          LOG_WARN("failed to finish pk expr", K(ret));
        }
      }
    }
  }
  return ret;
}

// Creates (or re-targets) tasks for every row whose AI column is still
// pending.  Called inside the user transaction right after the DML statement
// finished writing rows.
inline int build_ai_column_task_sweep_insert_sql(
    const common::ObString &db_name,
    const common::ObString &table_name,
    uint64_t table_id,
    uint64_t column_id,
    const common::ObString &status_column_name,
    const common::ObString &pk_expr,
    int64_t request_id,
    common::ObSqlString &out)
{
  return out.assign_fmt(
      "INSERT INTO oceanbase.__all_ai_column_task "
      "(table_id, column_id, rowkey_predicate, request_id, state, retry_count, eligible_ts) "
      "SELECT %lu, %lu, %.*s, %ld, 0, 0, now(6) "
      "FROM `%.*s`.`%.*s` WHERE `%.*s` = 0 "
      "ON DUPLICATE KEY UPDATE "
      "request_id = IF(state IN (2, 3), request_id, VALUES(request_id)), "
      "state = IF(state IN (2, 3), state, 0), "
      "retry_count = IF(state IN (2, 3), retry_count, 0), "
      "eligible_ts = IF(state IN (2, 3), eligible_ts, now(6))",
      static_cast<unsigned long>(table_id), static_cast<unsigned long>(column_id),
      pk_expr.length(), pk_expr.ptr(),
      static_cast<long>(request_id),
      db_name.length(), db_name.ptr(), table_name.length(), table_name.ptr(),
      status_column_name.length(), status_column_name.ptr());
}

// Removes tasks of rows that no longer exist (deleted or moved primary keys).
inline int build_ai_column_task_sweep_delete_sql(
    const common::ObString &db_name,
    const common::ObString &table_name,
    uint64_t table_id,
    uint64_t column_id,
    const common::ObString &first_rowkey_column_name,
    const common::ObString &pk_expr,
    common::ObSqlString &out)
{
  return out.assign_fmt(
      "DELETE task FROM oceanbase.__all_ai_column_task task "
      "LEFT JOIN `%.*s`.`%.*s` base ON task.rowkey_predicate = %.*s "
      "WHERE task.table_id = %lu AND task.column_id = %lu AND base.`%.*s` IS NULL",
      db_name.length(), db_name.ptr(), table_name.length(), table_name.ptr(),
      pk_expr.length(), pk_expr.ptr(),
      static_cast<unsigned long>(table_id), static_cast<unsigned long>(column_id),
      first_rowkey_column_name.length(), first_rowkey_column_name.ptr());
}

} // end namespace share
} // end namespace oceanbase

#endif // OCEANBASE_SHARE_SCHEMA_OB_AI_COLUMN_COMMON_H_
