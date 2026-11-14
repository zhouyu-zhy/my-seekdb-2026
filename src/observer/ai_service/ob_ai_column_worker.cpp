// [competition] ob_ai_column_worker.cpp
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

#define USING_LOG_PREFIX SERVER
#include "ob_ai_column_worker.h"

#include <unistd.h>
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "common/mysqlclient/ob_mysql_result.h"
#include "lib/allocator/page_arena.h"
#include "lib/string/ob_sql_string.h"
#include "share/ob_server_struct.h"
#include "share/rc/ob_server_runtime.h"
#include "share/schema/ob_ai_column_common.h"
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "observer/omt/ob_ai_service.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

namespace oceanbase
{
using namespace common;
using namespace share;
using namespace share::schema;
namespace observer
{

namespace
{
constexpr int64_t AI_COLUMN_WORKER_ROWS_PER_SCAN = 64;
constexpr int64_t AI_COLUMN_WORKER_IDLE_SLEEP_MS = 100;
constexpr int64_t AI_COLUMN_WORKER_RECONCILE_ROUNDS = 50;
constexpr int64_t AI_COLUMN_WORKER_STALE_CLAIM_SECONDS = 60;

int append_identifier(ObSqlString &sql, const ObString &name)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(sql.append("`"))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < name.length(); ++i) {
      if (name[i] == '`') {
        ret = sql.append("``", 2);
      } else {
        ret = sql.append(name.ptr() + i, 1);
      }
    }
    if (OB_SUCC(ret)) {
      ret = sql.append("`");
    }
  }
  return ret;
}

int append_qualified_table(ObSqlString &sql, const ObString &db_name, const ObString &table_name)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(append_identifier(sql, db_name))) {
  } else if (OB_FAIL(sql.append("."))) {
  } else if (OB_FAIL(append_identifier(sql, table_name))) {
  }
  return ret;
}

int obj_to_sql_literal(ObObj &obj, ObSqlString &sql, ObIAllocator &allocator)
{
  int ret = OB_SUCCESS;
  if (obj.is_null()) {
    ret = sql.append("NULL");
  } else {
    const int64_t buf_len = 1024 * 1024;
    char *buf = static_cast<char *>(allocator.alloc(buf_len));
    int64_t pos = 0;
    ObObjPrintParams print_params;
    if (OB_ISNULL(buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(obj.print_sql_literal(buf, buf_len, pos, print_params))) {
      if (OB_SIZE_OVERFLOW == ret) {
        ret = OB_BUF_NOT_ENOUGH;
      }
    } else if (OB_FAIL(sql.append(buf, pos))) {
    }
  }
  return ret;
}

// One claimed (table, rowkey_predicate, request_id) work unit.  All AI columns
// of that row share the request id, so they are evaluated and published in one
// step and become visible together.
struct AiTaskGroup
{
  uint64_t table_id_;
  int64_t request_id_;
  ObString predicate_;         // copied into the scan allocator
  ObSEArray<uint64_t, 4> column_ids_;
  ObSEArray<int64_t, 4> retry_counts_;
};

int group_key_eq(const AiTaskGroup &group, uint64_t table_id,
                 const ObString &predicate, int64_t request_id)
{
  return group.table_id_ == table_id
      && group.request_id_ == request_id
      && 0 == group.predicate_.compare(predicate);
}

int find_or_add_group(ObIArray<AiTaskGroup> &groups, ObArenaAllocator &alloc,
                      uint64_t table_id, const ObString &predicate,
                      int64_t request_id, int64_t &idx)
{
  int ret = OB_SUCCESS;
  idx = -1;
  for (int64_t i = 0; i < groups.count() && idx < 0; ++i) {
    if (group_key_eq(groups.at(i), table_id, predicate, request_id)) {
      idx = i;
    }
  }
  if (idx < 0) {
    AiTaskGroup group;
    group.table_id_ = table_id;
    group.request_id_ = request_id;
    char *buf = NULL;
    if (OB_ISNULL(buf = static_cast<char *>(alloc.alloc(predicate.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(buf, predicate.ptr(), predicate.length());
      group.predicate_.assign_ptr(buf, predicate.length());
      if (OB_FAIL(groups.push_back(group))) {
        LOG_WARN("failed to push task group", K(ret));
      } else {
        idx = groups.count() - 1;
      }
    }
  }
  return ret;
}

// WHERE clause locating every task row of the group inside the inner tables.
int append_group_condition(ObSqlString &sql, const AiTaskGroup &group)
{
  int ret = OB_SUCCESS;
  ObSqlString predicate_literal;
  if (OB_FAIL(sql.append_fmt("table_id = %lu AND request_id = %ld AND rowkey_predicate = ",
                             static_cast<unsigned long>(group.table_id_),
                             static_cast<long>(group.request_id_)))) {
    LOG_WARN("failed to append group condition", K(ret));
  } else if (OB_FAIL(build_ai_column_sql_literal(group.predicate_, predicate_literal))) {
    LOG_WARN("failed to quote predicate", K(ret));
  } else if (OB_FAIL(sql.append(predicate_literal.ptr()))) {
    LOG_WARN("failed to append predicate literal", K(ret));
  }
  return ret;
}

int append_column_id_list(ObSqlString &sql, const ObIArray<uint64_t> &column_ids)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < column_ids.count(); ++i) {
    if (i > 0 && OB_FAIL(sql.append(", "))) {
    } else if (OB_FAIL(sql.append_fmt("%lu", static_cast<unsigned long>(column_ids.at(i))))) {
    }
  }
  return ret;
}

// A value read through the sql proxy only lives as long as the result set it
// came from.  Anything needed after that result is closed has to be moved into
// a longer-lived allocator first, otherwise the text (prompt or embedding input)
// is read from memory the proxy already recycled.
int save_result_string(ObIAllocator &allocator, const ObString &row_value, ObString &saved)
{
  int ret = OB_SUCCESS;
  if (row_value.empty()) {
    saved.reset();
  } else {
    char *buf = NULL;
    if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(row_value.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to allocate ai column text buffer", K(ret), K(row_value));
    } else {
      MEMCPY(buf, row_value.ptr(), row_value.length());
      saved.assign_ptr(buf, static_cast<ObString::obstr_size_t>(row_value.length()));
    }
  }
  return ret;
}

// The completion transport returns the message as a JSON string node, so the
// payload arrives double-quoted.  AI columns store the bare text; peel one
// JSON string layer (with unescaping) before publishing.
int unwrap_json_string_value(ObIAllocator &allocator, const ObString &in, ObString &out)
{
  int ret = OB_SUCCESS;
  out = in;
  if (in.length() >= 2 && '"' == in[0] && '"' == in[in.length() - 1]) {
    char *buf = NULL;
    int64_t pos = 0;
    if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(in.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      for (int64_t i = 1; OB_SUCC(ret) && i < in.length() - 1; ++i) {
        char c = in[i];
        if ('\\' != c) {
          buf[pos++] = c;
        } else if (++i >= in.length() - 1) {
          ret = OB_INVALID_DATA;
        } else {
          const char esc = in[i];
          char plain = esc;
          switch (esc) {
            case 'n': plain = '\n'; break;
            case 't': plain = '\t'; break;
            case 'r': plain = '\r'; break;
            case 'b': plain = '\b'; break;
            case 'f': plain = '\f'; break;
            case '"': plain = '"'; break;
            case '\\': plain = '\\'; break;
            case '/': plain = '/'; break;
            default:
              plain = esc;
              break;
          }
          buf[pos++] = plain;
        }
      }
      if (OB_SUCC(ret)) {
        out.assign_ptr(buf, pos);
      }
    }
  }
  return ret;
}

// Runs the model call for one AI column on the current row values.  Completion
// columns carry the type-derived system prompt; embedding columns do not.
// A non-empty system prompt marks the request as AI-column generated so mock
// and real endpoints can return typed payloads.
int evaluate_ai_column_value(const ObColumnSchemaV2 &ai_column,
                             const ObString &db_name,
                             const ObString &table_name,
                             const ObSqlString &pk_where,
                             ObIAllocator &allocator,
                             bool &evaluated,
                             ObSqlString &value_literal)
{
  int ret = OB_SUCCESS;
  evaluated = false;
  const ObString embed_model = ai_column.get_ai_column_embed_model();
  const ObString embed_input = ai_column.get_ai_column_embed_input();
  const ObString complete_model = ai_column.get_ai_column_complete_model();
  const ObString complete_prompt = ai_column.get_ai_column_complete_prompt();
  if (!complete_model.empty() && !complete_prompt.empty()) {
    ObSqlString prompt_query;
    ObString prompt_text;
    if (OB_FAIL(prompt_query.append("SELECT ("))) {
    } else if (OB_FAIL(prompt_query.append(complete_prompt.ptr(), complete_prompt.length()))) {
    } else if (OB_FAIL(prompt_query.append(") FROM "))) {
    } else if (OB_FAIL(append_qualified_table(prompt_query, db_name, table_name))) {
    } else if (OB_FAIL(prompt_query.append(" WHERE "))) {
    } else if (OB_FAIL(prompt_query.append(pk_where.ptr()))) {
    }
    if (OB_SUCC(ret)) {
      SMART_VAR(ObMySQLProxy::MySQLResult, prompt_res) {
        ObObj prompt_obj;
        if (OB_FAIL(GCTX.sql_proxy_->read(prompt_res, prompt_query.ptr()))) {
        } else if (OB_ISNULL(prompt_res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_ITER_END == (ret = prompt_res.get_result()->next())) {
          ret = OB_ENTRY_NOT_EXIST;
        } else if (OB_FAIL(ret)) {
        } else if (OB_FAIL(prompt_res.get_result()->get_obj(static_cast<int64_t>(0), prompt_obj))) {
        } else if (prompt_obj.is_null()) {
          ret = value_literal.append("NULL");
          evaluated = OB_SUCC(ret);
        } else if (OB_FAIL(save_result_string(allocator, prompt_obj.get_string(), prompt_text))) {
          LOG_WARN("failed to keep ai column prompt text", K(ret));
        } else if (prompt_text.empty()) {
          ret = value_literal.append("NULL");
          evaluated = OB_SUCC(ret);
        }
      }
    }
    if (OB_SUCC(ret) && !evaluated) {
      ObAIFuncExprInfo *info = NULL;
      const ObAiModelEndpointInfo *endpoint = NULL;
      omt::ObAiServiceGuard guard;
      if (OB_FAIL(guard.get_ai_endpoint_by_ai_model_name(complete_model, endpoint, false))) {
      } else if (OB_ISNULL(endpoint)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, complete_model, info))) {
      } else if (OB_ISNULL(info)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObAIFuncModel model(allocator, *info, *endpoint);
        ObString result;
        const char *prompt_chars = get_ai_column_system_prompt(ai_column);
        ObString system_prompt;
        if (OB_NOT_NULL(prompt_chars)) {
          system_prompt = ObString::make_string(prompt_chars);
        }
        if (system_prompt.empty()) {
          ret = model.call_completion(prompt_text, NULL, result);
        } else {
          ret = model.call_completion(system_prompt, prompt_text, NULL, result);
        }
        if (OB_FAIL(ret)) {
          LOG_WARN("ai column completion call failed", K(ret),
                   "column_id", ai_column.get_column_id());
        } else {
          ObString plain;
          ObObj result_obj;
          if (OB_FAIL(unwrap_json_string_value(allocator, result, plain))) {
            LOG_WARN("failed to unwrap completion payload", K(ret));
          } else {
            result_obj.set_varchar(plain);
            result_obj.set_collation_type(CS_TYPE_UTF8MB4_BIN);
            result_obj.set_collation_level(CS_LEVEL_IMPLICIT);
            if (OB_FAIL(obj_to_sql_literal(result_obj, value_literal, allocator))) {
            } else {
              evaluated = true;
            }
          }
        }
      }
    }
  } else if (!embed_model.empty() && !embed_input.empty()) {
    ObSqlString input_query;
    ObString input_text;
    if (OB_FAIL(input_query.append("SELECT ("))) {
    } else if (OB_FAIL(input_query.append(embed_input.ptr(), embed_input.length()))) {
    } else if (OB_FAIL(input_query.append(") FROM "))) {
    } else if (OB_FAIL(append_qualified_table(input_query, db_name, table_name))) {
    } else if (OB_FAIL(input_query.append(" WHERE "))) {
    } else if (OB_FAIL(input_query.append(pk_where.ptr()))) {
    }
    if (OB_SUCC(ret)) {
      SMART_VAR(ObMySQLProxy::MySQLResult, input_res) {
        ObObj input_obj;
        if (OB_FAIL(GCTX.sql_proxy_->read(input_res, input_query.ptr()))) {
        } else if (OB_ISNULL(input_res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_ITER_END == (ret = input_res.get_result()->next())) {
          ret = OB_ENTRY_NOT_EXIST;
        } else if (OB_FAIL(ret)) {
        } else if (OB_FAIL(input_res.get_result()->get_obj(static_cast<int64_t>(0), input_obj))) {
        } else if (input_obj.is_null()) {
          ret = value_literal.append("NULL");
          evaluated = OB_SUCC(ret);
        } else if (OB_FAIL(save_result_string(allocator, input_obj.get_string(), input_text))) {
          LOG_WARN("failed to keep ai column embedding input", K(ret));
        } else if (input_text.empty()) {
          ret = value_literal.append("NULL");
          evaluated = OB_SUCC(ret);
        }
      }
    }
    if (OB_SUCC(ret) && !evaluated) {
      ObAIFuncExprInfo *info = NULL;
      const ObAiModelEndpointInfo *endpoint = NULL;
      omt::ObAiServiceGuard guard;
      if (OB_FAIL(guard.get_ai_endpoint_by_ai_model_name(embed_model, endpoint, false))) {
      } else if (OB_ISNULL(endpoint)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, embed_model, info))) {
      } else if (OB_ISNULL(info)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObAIFuncModel model(allocator, *info, *endpoint);
        ObJsonObject *config = NULL;
        ObJsonInt *dim_json = NULL;
        ObString result;
        int64_t dimension = ai_column.get_ai_column_embed_dimension();
        if (dimension <= 0) {
          dimension = 4;
        }
        if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config))) {
        } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, dimension, dim_json))) {
        } else if (OB_FAIL(config->add("dimensions", dim_json))) {
        } else if (OB_FAIL(model.call_dense_embedding(input_text, config, result))) {
        } else {
          ObObj result_obj;
          result_obj.set_varchar(result);
          result_obj.set_collation_type(CS_TYPE_UTF8MB4_BIN);
          result_obj.set_collation_level(CS_LEVEL_IMPLICIT);
          if (OB_FAIL(obj_to_sql_literal(result_obj, value_literal, allocator))) {
          } else {
            evaluated = true;
          }
        }
      }
    }
  } else {
    // Plain expression AI column: evaluate the stored expression against the row.
    ObSqlString eval_query;
    if (OB_FAIL(eval_query.append("SELECT ("))) {
    } else if (OB_FAIL(eval_query.append(ai_column.get_ai_column_expr().ptr(),
                                         ai_column.get_ai_column_expr().length()))) {
    } else if (OB_FAIL(eval_query.append(") FROM "))) {
    } else if (OB_FAIL(append_qualified_table(eval_query, db_name, table_name))) {
    } else if (OB_FAIL(eval_query.append(" WHERE "))) {
    } else if (OB_FAIL(eval_query.append(pk_where.ptr()))) {
    } else {
      SMART_VAR(ObMySQLProxy::MySQLResult, eval_res) {
        ObObj value_obj;
        if (OB_FAIL(GCTX.sql_proxy_->read(eval_res, eval_query.ptr()))) {
        } else if (OB_ISNULL(eval_res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_ITER_END == (ret = eval_res.get_result()->next())) {
          ret = OB_ENTRY_NOT_EXIST;
        } else if (OB_FAIL(ret)) {
        } else if (OB_FAIL(eval_res.get_result()->get_obj(static_cast<int64_t>(0), value_obj))) {
        } else if (OB_FAIL(obj_to_sql_literal(value_obj, value_literal, allocator))) {
        } else {
          evaluated = true;
        }
      }
    }
  }
  return ret;
}

// Deletes task rows of a group for one or all columns.
int delete_group_tasks(const AiTaskGroup &group, const ObIArray<uint64_t> *column_ids)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_FAIL(sql.append("DELETE FROM oceanbase.__all_ai_column_task WHERE "))) {
    LOG_WARN("failed to start task delete", K(ret));
  } else if (OB_FAIL(append_group_condition(sql, group))) {
  } else if (OB_NOT_NULL(column_ids) && !column_ids->empty()) {
    if (OB_FAIL(sql.append(" AND column_id IN ("))) {
    } else if (OB_FAIL(append_column_id_list(sql, *column_ids))) {
    } else if (OB_FAIL(sql.append(")"))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(GCTX.sql_proxy_->write(sql.ptr(), affected_rows))) {
    LOG_WARN("failed to delete group tasks", K(ret), K(sql));
  }
  return ret;
}

// Moves a retryable failure back to state 3 with a short delay; retry_count is
// only bumped for the columns whose model call failed.
int defer_group_tasks(const AiTaskGroup &group, const ObIArray<uint64_t> &failed_column_ids)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_FAIL(sql.append("UPDATE oceanbase.__all_ai_column_task "
                         "SET state = 3, "
                         "retry_count = IF(column_id IN ("))) {
    LOG_WARN("failed to start task defer", K(ret));
  } else if (OB_FAIL(append_column_id_list(sql, failed_column_ids))) {
  } else if (OB_FAIL(sql.append("), retry_count + 1, retry_count), "
                               "eligible_ts = now(6) + interval 1 second WHERE "))) {
  } else if (OB_FAIL(append_group_condition(sql, group))) {
  } else if (OB_FAIL(GCTX.sql_proxy_->write(sql.ptr(), affected_rows))) {
    LOG_WARN("failed to defer group tasks", K(ret), K(sql));
  }
  return ret;
}

// A stale publish (row changed underneath us) must not lose the task: reopen
// it so the next claim works on the fresh generation.
int reopen_group_tasks(const AiTaskGroup &group)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_FAIL(sql.append("UPDATE oceanbase.__all_ai_column_task "
                         "SET state = 0, eligible_ts = now(6) WHERE "))) {
    LOG_WARN("failed to start task reopen", K(ret));
  } else if (OB_FAIL(append_group_condition(sql, group))) {
  } else if (OB_FAIL(GCTX.sql_proxy_->write(sql.ptr(), affected_rows))) {
    LOG_WARN("failed to reopen group tasks", K(ret), K(sql));
  }
  return ret;
}

// Processes one claimed group: reads the row snapshot, evaluates every column,
// then publishes all-or-nothing with a generation guard.
int process_claimed_group(const ObTableSchema &table_schema,
                          const ObString &db_name,
                          const AiTaskGroup &group)
{
  int ret = OB_SUCCESS;
  ObArenaAllocator allocator("AiColGroup");
  const ObString &table_name = table_schema.get_table_name_str();
  ObSqlString pk_where;
  struct ColWork
  {
    const ObColumnSchemaV2 *ai_column_;
    const ObColumnSchemaV2 *status_column_;
    const ObColumnSchemaV2 *gen_column_;
    int64_t task_retry_count_;
    int64_t status_value_;
    int64_t gen_value_;
    bool active_;               // row status was pending at snapshot time
    bool failed_;               // model call failed and no retry budget left
    bool needs_retry_;
    ObString value_;            // publish literal, copied into the arena
  };
  ObSEArray<ColWork, 4> works;
  for (int64_t i = 0; OB_SUCC(ret) && i < group.column_ids_.count(); ++i) {
    const ObColumnSchemaV2 *ai_column = table_schema.get_column_schema(group.column_ids_.at(i));
    ColWork work;
    work.ai_column_ = ai_column;
    work.status_column_ = NULL;
    work.gen_column_ = NULL;
    work.task_retry_count_ = group.retry_counts_.at(i);
    work.status_value_ = OB_AI_COLUMN_STATUS_PENDING;
    work.gen_value_ = 0;
    work.active_ = false;
    work.failed_ = false;
    work.needs_retry_ = false;
    if (OB_ISNULL(ai_column) || !ai_column->is_ai_column()) {
      // column dropped by an ALTER while the task was queued
      ObSEArray<uint64_t, 1> stale_ids;
      if (OB_FAIL(stale_ids.push_back(group.column_ids_.at(i)))) {
        LOG_WARN("failed to push stale column id", K(ret));
      } else if (OB_FAIL(delete_group_tasks(group, &stale_ids))) {
        LOG_WARN("failed to remove stale task", K(ret));
      }
    } else if (OB_ISNULL(work.status_column_ = table_schema.get_column_schema(
                   ai_column->get_ai_column_status_column_id()))) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("ai column without status column", K(ret), "column_id", ai_column->get_column_id());
    } else if (OB_ISNULL(work.gen_column_ = table_schema.get_column_schema(
                   ai_column->get_ai_column_gen_column_id()))) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("ai column without gen column", K(ret), "column_id", ai_column->get_column_id());
    } else if (OB_FAIL(works.push_back(work))) {
      LOG_WARN("failed to push col work", K(ret));
    }
  }
  if (OB_FAIL(ret) || works.empty()) {
  } else if (OB_FAIL(pk_where.append(group.predicate_.ptr(), group.predicate_.length()))) {
    LOG_WARN("failed to copy predicate", K(ret));
  } else {
    // Snapshot read: status and generation per column plus a row existence check.
    ObSqlString snapshot_query;
    if (OB_FAIL(snapshot_query.append("SELECT "))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
      if (i > 0 && OB_FAIL(snapshot_query.append(", "))) {
      } else if (OB_FAIL(append_identifier(snapshot_query,
                                           works.at(i).status_column_->get_column_name_str()))) {
      } else if (OB_FAIL(snapshot_query.append(", "))) {
      } else if (OB_FAIL(append_identifier(snapshot_query,
                                           works.at(i).gen_column_->get_column_name_str()))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(snapshot_query.append(" FROM "))) {
    } else if (OB_FAIL(append_qualified_table(snapshot_query, db_name, table_name))) {
    } else if (OB_FAIL(snapshot_query.append(" WHERE "))) {
    } else if (OB_FAIL(snapshot_query.append(pk_where.ptr()))) {
    } else {
      bool row_found = false;
      SMART_VAR(ObMySQLProxy::MySQLResult, snapshot_res) {
        if (OB_FAIL(GCTX.sql_proxy_->read(snapshot_res, snapshot_query.ptr()))) {
        } else if (OB_ISNULL(snapshot_res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          sqlclient::ObMySQLResult *result = snapshot_res.get_result();
          int read_ret = result->next();
          if (OB_ITER_END == read_ret) {
            row_found = false;
          } else if (OB_SUCCESS != read_ret) {
            ret = read_ret;
            LOG_WARN("failed to read snapshot row", K(ret));
          } else {
            row_found = true;
            for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
              int64_t status_value = 0;
              int64_t gen_value = 0;
              if (OB_FAIL(result->get_int(static_cast<int64_t>(2 * i), status_value))) {
                LOG_WARN("failed to read status value", K(ret), K(i));
              } else if (OB_FAIL(result->get_int(static_cast<int64_t>(2 * i + 1), gen_value))) {
                LOG_WARN("failed to read gen value", K(ret), K(i));
              } else {
                works.at(i).status_value_ = status_value;
                works.at(i).gen_value_ = gen_value;
                works.at(i).active_ = (OB_AI_COLUMN_STATUS_PENDING == status_value);
              }
            }
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (!row_found) {
        // row deleted while the group was claimed
        if (OB_FAIL(delete_group_tasks(group, NULL))) {
          LOG_WARN("failed to drop tasks of missing row", K(ret));
        }
      } else {
        // evaluate the still-pending columns
        for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
          if (!works.at(i).active_) {
            continue;
          }
          works.at(i).value_.reset();
          ObSqlString value_literal;
          bool evaluated = false;
          int eval_ret = evaluate_ai_column_value(*works.at(i).ai_column_, db_name, table_name,
                                                  pk_where, allocator, evaluated, value_literal);
          if (OB_SUCCESS == eval_ret && evaluated) {
            char *buf = NULL;
            if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(value_literal.length())))) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else {
              MEMCPY(buf, value_literal.ptr(), value_literal.length());
              works.at(i).value_.assign_ptr(buf, value_literal.length());
            }
          }
          if (OB_SUCCESS != eval_ret || !evaluated) {
            const int64_t max_retries = works.at(i).ai_column_->get_ai_column_max_retries();
            if (works.at(i).task_retry_count_ < max_retries) {
              works.at(i).needs_retry_ = true;
            } else {
              works.at(i).failed_ = true;
            }
            LOG_WARN("ai column evaluation failed", K(eval_ret),
                     "table_id", group.table_id_,
                     "column_id", works.at(i).ai_column_->get_column_id(),
                     "retry_count", works.at(i).task_retry_count_,
                     "max_retries", max_retries);
          }
        }
        // Drop tasks whose row is no longer pending; they were resolved elsewhere.
        ObSEArray<uint64_t, 4> idle_ids;
        for (int64_t i = 0; i < works.count(); ++i) {
          if (!works.at(i).active_) {
            int tmp_ret = idle_ids.push_back(works.at(i).ai_column_->get_column_id());
            UNUSED(tmp_ret);
          }
        }
        if (OB_SUCC(ret) && !idle_ids.empty() && OB_FAIL(delete_group_tasks(group, &idle_ids))) {
          LOG_WARN("failed to remove resolved tasks", K(ret));
        }
        bool any_work = false;
        for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
          any_work |= works.at(i).active_;
        }
        if (OB_FAIL(ret) || !any_work) {
        } else {
          bool need_defer = false;
          for (int64_t i = 0; i < works.count(); ++i) {
            need_defer |= (works.at(i).active_ && works.at(i).needs_retry_);
          }
          if (need_defer) {
            ObSEArray<uint64_t, 4> failed_ids;
            for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
              if (works.at(i).active_ && works.at(i).needs_retry_
                  && OB_FAIL(failed_ids.push_back(works.at(i).ai_column_->get_column_id()))) {
                LOG_WARN("failed to push failed column id", K(ret));
              }
            }
            if (OB_SUCC(ret) && OB_FAIL(defer_group_tasks(group, failed_ids))) {
              LOG_WARN("failed to defer retryable group", K(ret));
            }
          } else {
            // Single guarded publish: every column flips to READY/FAILED
            // atomically, so readers never observe a half-generated row.
            ObSqlString update;
            bool has_set = false;
            if (OB_FAIL(update.append("UPDATE "))) {
            } else if (OB_FAIL(append_qualified_table(update, db_name, table_name))) {
            } else if (OB_FAIL(update.append(" SET "))) {
            }
            for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
              if (!works.at(i).active_) {
                continue;
              }
              const bool ok = !works.at(i).failed_;
              if (has_set && OB_FAIL(update.append(", "))) {
              } else if (OB_FAIL(append_identifier(update,
                                                   works.at(i).ai_column_->get_column_name_str()))) {
              } else if (OB_FAIL(update.append(" = "))) {
              } else if (ok) {
                if (OB_FAIL(update.append(works.at(i).value_.ptr(), works.at(i).value_.length()))) {
                  LOG_WARN("failed to append value literal", K(ret));
                }
              } else if (OB_FAIL(update.append("NULL"))) {
              }
              if (OB_FAIL(ret)) {
              } else if (OB_FAIL(update.append(", "))) {
              } else if (OB_FAIL(append_identifier(update,
                                        works.at(i).status_column_->get_column_name_str()))) {
              } else if (OB_FAIL(update.append_fmt(" = %ld", ok ? 1 : 2))) {
              } else {
                has_set = true;
              }
            }
            if (OB_FAIL(ret)) {
            } else if (!has_set) {
              // nothing left to publish
            } else if (OB_FAIL(update.append(" WHERE "))) {
            } else if (OB_FAIL(update.append(pk_where.ptr()))) {
            }
            for (int64_t i = 0; OB_SUCC(ret) && i < works.count(); ++i) {
              if (!works.at(i).active_) {
                continue;
              }
              if (OB_FAIL(update.append(" AND "))) {
              } else if (OB_FAIL(append_identifier(update,
                                        works.at(i).gen_column_->get_column_name_str()))) {
              } else if (OB_FAIL(update.append_fmt(" = %ld", works.at(i).gen_value_))) {
              }
            }
            if (OB_FAIL(ret)) {
            } else if (!has_set) {
            } else {
              int64_t affected_rows = 0;
              if (OB_FAIL(GCTX.sql_proxy_->write(update.ptr(), affected_rows))) {
                LOG_WARN("failed to publish ai column result", K(ret), K(update));
              } else if (affected_rows > 0) {
                if (OB_FAIL(delete_group_tasks(group, NULL))) {
                  LOG_WARN("failed to delete completed tasks", K(ret));
                }
              } else if (OB_FAIL(reopen_group_tasks(group))) {
                LOG_WARN("failed to reopen stale tasks", K(ret));
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

// Backstop pass: reopen claims abandoned by a crashed worker and enqueue rows
// whose task creation was missed (status still pending without a task row).
int reconcile_ai_column_tasks(const ObTableSchema &table_schema, const ObString &db_name)
{
  int ret = OB_SUCCESS;
  static int64_t reconcile_request_id = 0;
  int64_t affected_rows = 0;
  ObSqlString reset_sql;
  if (OB_FAIL(reset_sql.append_fmt("UPDATE oceanbase.__all_ai_column_task "
                                   "SET state = 0, eligible_ts = now(6) "
                                   "WHERE table_id = %lu AND state = 2 "
                                   "AND gmt_modified < now(6) - interval %ld second",
                                   static_cast<unsigned long>(table_schema.get_table_id()),
                                   static_cast<long>(AI_COLUMN_WORKER_STALE_CLAIM_SECONDS)))) {
    LOG_WARN("failed to build stale claim reset", K(ret));
  } else if (OB_FAIL(GCTX.sql_proxy_->write(reset_sql.ptr(), affected_rows))) {
    LOG_WARN("failed to reset stale claims", K(ret), K(reset_sql));
  }
  if (OB_SUCC(ret)) {
    bool has_ai_column = false;
    ObString first_rowkey_name;
    for (ObTableSchema::const_column_iterator iter = table_schema.column_begin();
         iter != table_schema.column_end(); ++iter) {
      const ObColumnSchemaV2 *column = *iter;
      if (OB_NOT_NULL(column)) {
        if (column->is_ai_column()) {
          has_ai_column = true;
        } else if (column->is_rowkey_column()
                   && first_rowkey_name.empty()) {
          first_rowkey_name = column->get_column_name_str();
        }
      }
    }
    if (has_ai_column && !first_rowkey_name.empty()) {
      const int64_t request_id = ATOMIC_AAF(&reconcile_request_id, 1);
      ObSqlString pk_expr;
      if (OB_FAIL(build_ai_column_pk_expr(table_schema, pk_expr))) {
        LOG_WARN("failed to build pk expr", K(ret));
      }
      for (ObTableSchema::const_column_iterator iter = table_schema.column_begin();
           OB_SUCC(ret) && iter != table_schema.column_end(); ++iter) {
        const ObColumnSchemaV2 *column = *iter;
        if (OB_NOT_NULL(column) && column->is_ai_column()) {
          const ObColumnSchemaV2 *status_column = table_schema.get_column_schema(
              column->get_ai_column_status_column_id());
          if (OB_ISNULL(status_column)) {
            // incompatible column definition; leave it to user DML
          } else {
            ObSqlString insert_sql;
            ObSqlString delete_sql;
            if (OB_FAIL(build_ai_column_task_sweep_insert_sql(
                    db_name, table_schema.get_table_name_str(),
                    table_schema.get_table_id(), column->get_column_id(),
                    status_column->get_column_name_str(), pk_expr.string(),
                    request_id, insert_sql))) {
              LOG_WARN("failed to build reconcile sweep", K(ret));
            } else if (OB_FAIL(GCTX.sql_proxy_->write(insert_sql.ptr(), affected_rows))) {
              LOG_WARN("failed to run reconcile sweep", K(ret), K(insert_sql));
            } else if (OB_FAIL(build_ai_column_task_sweep_delete_sql(
                    db_name, table_schema.get_table_name_str(),
                    table_schema.get_table_id(), column->get_column_id(),
                    first_rowkey_name, pk_expr.string(), delete_sql))) {
              LOG_WARN("failed to build reconcile cleanup", K(ret));
            } else if (OB_FAIL(GCTX.sql_proxy_->write(delete_sql.ptr(), affected_rows))) {
              LOG_WARN("failed to run reconcile cleanup", K(ret), K(delete_sql));
            }
          }
        }
      }
    }
  }
  return ret;
}

} // namespace

ObAiColumnWorker::ObAiColumnWorker()
  : is_inited_(false),
    stop_(true),
    thread_()
{
}

ObAiColumnWorker::~ObAiColumnWorker()
{
  destroy();
}

int ObAiColumnWorker::init()
{
  int ret = OB_SUCCESS;
  if (is_inited_) {
    ret = OB_INIT_TWICE;
  } else {
    is_inited_ = true;
  }
  return ret;
}

int ObAiColumnWorker::start()
{
  int ret = OB_SUCCESS;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else if (!stop_) {
    ret = OB_INIT_TWICE;
  } else {
    stop_ = false;
    thread_ = std::thread([this]() { run_(); });
  }
  return ret;
}

void ObAiColumnWorker::stop()
{
  stop_ = true;
}

void ObAiColumnWorker::wait()
{
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ObAiColumnWorker::destroy()
{
  stop();
  wait();
  is_inited_ = false;
}

void ObAiColumnWorker::run_()
{
  int ret = OB_SUCCESS;
  while (!stop_) {
    if (OB_FAIL(process_once_())) {
      if (OB_SUCCESS != ret) {
        LOG_WARN("process ai column once failed", KR(ret));
      }
    }
    // One round per idle interval.  The interval has to stay well below the
    // gap a user leaves between two writes to the same row, otherwise the
    // second write rewrites the pending task before the worker ever claimed
    // the first one, and the "late result of an obsolete request" guard has
    // nothing to protect.  Sliced sleeps keep stop_ responsive.
    for (int64_t i = 0; !stop_ && i < AI_COLUMN_WORKER_IDLE_SLEEP_MS / 10; ++i) {
      usleep(10 * 1000);
    }
  }
}

int ObAiColumnWorker::process_once_()
{
  int ret = OB_SUCCESS;
  static int64_t round_counter = 0;
  ObSchemaGetterGuard guard;
  if (OB_ISNULL(GCTX.schema_service_) || OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(guard))) {
    LOG_WARN("failed to get runtime schema guard", K(ret));
  } else {
    const bool reconcile_round
        = (0 == ++round_counter % AI_COLUMN_WORKER_RECONCILE_ROUNDS);
    ObArenaAllocator scan_alloc("AiColScan");
    ObSEArray<AiTaskGroup, 8> groups;
    ObSqlString fetch_sql;
    if (OB_FAIL(fetch_sql.append(
            "SELECT table_id, column_id, rowkey_predicate, request_id, retry_count "
            "FROM oceanbase.__all_ai_column_task "
            "WHERE state IN (0, 3) AND eligible_ts <= now(6) "
            "ORDER BY table_id, rowkey_predicate, request_id, column_id LIMIT "))) {
      LOG_WARN("failed to build fetch sql", K(ret));
    } else if (OB_FAIL(fetch_sql.append_fmt("%ld", AI_COLUMN_WORKER_ROWS_PER_SCAN))) {
    } else {
      SMART_VAR(ObMySQLProxy::MySQLResult, fetch_res) {
        if (OB_FAIL(GCTX.sql_proxy_->read(fetch_res, fetch_sql.ptr()))) {
          LOG_WARN("failed to fetch ai tasks", K(ret));
        } else if (OB_ISNULL(fetch_res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          sqlclient::ObMySQLResult *result = fetch_res.get_result();
          int read_ret = OB_SUCCESS;
          // OB_SUCC() assigns its argument to ret, which would leak the final
          // OB_ITER_END out of this round and skip the claim stage entirely;
          // compare against the constant instead.
          while (OB_SUCCESS == read_ret && !stop_) {
            read_ret = result->next();
            if (OB_SUCCESS != read_ret) {
              if (OB_ITER_END != read_ret) {
                ret = read_ret;
                LOG_WARN("failed to walk ai tasks", K(ret));
              }
            } else {
              int64_t table_id = 0;
              int64_t column_id = 0;
              int64_t request_id = 0;
              int64_t retry_count = 0;
              ObObj predicate_obj;
              int64_t idx = -1;
              if (OB_FAIL(result->get_int(static_cast<int64_t>(0), table_id))) {
                LOG_WARN("failed to read table_id", K(ret));
              } else if (OB_FAIL(result->get_int(static_cast<int64_t>(1), column_id))) {
                LOG_WARN("failed to read column_id", K(ret));
              } else if (OB_FAIL(result->get_obj(static_cast<int64_t>(2), predicate_obj))) {
                LOG_WARN("failed to read predicate", K(ret));
              } else if (OB_FAIL(result->get_int(static_cast<int64_t>(3), request_id))) {
                LOG_WARN("failed to read request_id", K(ret));
              } else if (OB_FAIL(result->get_int(static_cast<int64_t>(4), retry_count))) {
                LOG_WARN("failed to read retry_count", K(ret));
              } else {
                ObString predicate = predicate_obj.get_string();
                if (OB_FAIL(find_or_add_group(groups, scan_alloc,
                                              static_cast<uint64_t>(table_id),
                                              predicate, request_id, idx))) {
                  LOG_WARN("failed to group task", K(ret));
                } else if (idx >= 0) {
                  int tmp_ret = OB_SUCCESS;
                  if (OB_SUCCESS != (tmp_ret = groups.at(idx).column_ids_.push_back(
                          static_cast<uint64_t>(column_id)))) {
                    // group array is best-effort sized; skip overflow
                  } else if (OB_SUCCESS != (tmp_ret = groups.at(idx).retry_counts_.push_back(retry_count))) {
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < groups.count() && !stop_; ++i) {
      const AiTaskGroup &group = groups.at(i);
      ObSqlString claim_sql;
      ObSqlString predicate_literal;
      int64_t affected_rows = 0;
      if (OB_FAIL(build_ai_column_sql_literal(group.predicate_, predicate_literal))) {
        LOG_WARN("failed to quote predicate", K(ret));
      } else if (OB_FAIL(claim_sql.append_fmt(
                     "UPDATE oceanbase.__all_ai_column_task SET state = 2 "
                     "WHERE table_id = %lu AND request_id = %ld AND rowkey_predicate = %.*s "
                     "AND state IN (0, 3)",
                     static_cast<unsigned long>(group.table_id_),
                     static_cast<long>(group.request_id_),
                     predicate_literal.length(), predicate_literal.ptr()))) {
        LOG_WARN("failed to build claim sql", K(ret));
      } else if (OB_FAIL(GCTX.sql_proxy_->write(claim_sql.ptr(), affected_rows))) {
        LOG_WARN("failed to claim ai task group", K(ret), K(claim_sql));
        ret = OB_SUCCESS;  // try the next group
      } else if (affected_rows == 0) {
        // another worker owns this group already
      } else {
        const ObTableSchema *table_schema = NULL;
        const ObDatabaseSchema *db_schema = NULL;
        if (OB_FAIL(guard.get_table_schema(group.table_id_, table_schema))) {
          LOG_WARN("failed to get table schema", K(ret), "table_id", group.table_id_);
        } else if (OB_ISNULL(table_schema)) {
          // table dropped: the drop path removed metadata and tasks
          ret = OB_SUCCESS;
        } else if (OB_FAIL(guard.get_database_schema(table_schema->get_database_id(), db_schema))
                   || OB_ISNULL(db_schema)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          int proc_ret = process_claimed_group(*table_schema,
                                               db_schema->get_database_name_str(), group);
          if (OB_SUCCESS != proc_ret) {
            LOG_WARN("failed to process ai task group", K(proc_ret),
                     "table_id", group.table_id_, "request_id", group.request_id_);
          }
        }
      }
    }
    if (OB_SUCC(ret) && reconcile_round) {
      ObSEArray<uint64_t, 32> table_ids;
      if (OB_FAIL(guard.get_table_ids_in_runtime(table_ids))) {
        LOG_WARN("failed to get runtime tables", K(ret));
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < table_ids.count() && !stop_; ++i) {
        const ObTableSchema *table_schema = NULL;
        const ObDatabaseSchema *db_schema = NULL;
        bool has_ai_column = false;
        if (OB_FAIL(guard.get_table_schema(table_ids.at(i), table_schema))) {
        } else if (OB_ISNULL(table_schema) || !table_schema->is_user_table()) {
          // skip
        } else {
          for (ObTableSchema::const_column_iterator iter = table_schema->column_begin();
               iter != table_schema->column_end(); ++iter) {
            if (OB_NOT_NULL(*iter) && (*iter)->is_ai_column()) {
              has_ai_column = true;
              break;
            }
          }
          if (has_ai_column) {
            if (OB_FAIL(guard.get_database_schema(table_schema->get_database_id(), db_schema))
                || OB_ISNULL(db_schema)) {
              ret = OB_ERR_UNEXPECTED;
            } else {
              int tmp_ret = reconcile_ai_column_tasks(*table_schema,
                                                      db_schema->get_database_name_str());
              if (OB_SUCCESS != tmp_ret) {
                LOG_WARN("failed to reconcile ai column tasks", K(tmp_ret),
                         "table_id", table_schema->get_table_id());
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

} // namespace observer
} // end namespace oceanbase
