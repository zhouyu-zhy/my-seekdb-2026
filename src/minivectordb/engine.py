"""Execution engine: catalog, transactions, SQL execution and vector TopK.

Visibility model
----------------
``Table.live`` holds the committed ``{pk: (offset, length)}`` map rebuilt from
the table log.  A transaction keeps its own overlay so that it reads its own
writes without touching committed state.  Records reach the file immediately,
but nothing becomes visible until the commit record in ``commit.log`` is
fsynced.  Rollback and crash recovery therefore only have to ignore frames of
transactions that never committed; there is no undo image on disk.

Two query paths
---------------
``_select_columnar`` builds a bounded columnar snapshot and evaluates filters,
ordering and distances with numpy.  ``_select_rows`` is the plain row-by-row
reference path used when numpy is absent, when the snapshot would exceed
``MVDB_COLUMNAR_BUDGET`` or when a case cannot be vectorised.  Both share the
same preparation and emission rules, and the environment switch
``MVDB_DISABLE_COLUMNAR=1`` forces the reference path for cross-checking.
"""

import os

from . import errors, sqlparser, storage, values

try:  # optional acceleration, never required for correctness
    import numpy as _np
except Exception:  # pragma: no cover - exercised only without numpy
    _np = None

COLUMNAR_BUDGET = int(os.environ.get("MVDB_COLUMNAR_BUDGET", str(96 * 1024 * 1024)))
DISABLE_COLUMNAR = os.environ.get("MVDB_DISABLE_COLUMNAR", "") not in ("", "0")
ROW_CACHE_BUDGET = int(os.environ.get("MVDB_ROW_CACHE_BUDGET", str(32 * 1024 * 1024)))

COMPARATORS = {
    "=": lambda left, right: left == right,
    "<>": lambda left, right: left != right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
}

FLIPPED = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "<>": "<>"}


class _Absent:
    __slots__ = ()


_ABSENT = _Absent()


class _FallbackToRows(Exception):
    """Raised inside the columnar path when a case needs the reference path."""


class Table:
    def __init__(self, meta, data_dir, committed, create=False):
        self.meta = meta
        self.name = meta["name"]
        self.columns = meta["columns"]
        self.path = os.path.join(data_dir, meta["file"])
        self.log, self.live = storage.recover(self.path, committed, create=create)
        self.primary_index = meta["primary_index"]
        self.vector_index = next(
            (i for i, column in enumerate(self.columns) if column["type"] == "vector"), None
        )
        self.generation = 0
        self._rows = {}
        self._row_bytes = 0
        self._snapshot = None
        self._sorted = None

    # ------------------------------------------------------------------ cache
    def bump(self):
        self.generation += 1
        self._snapshot = None
        self._sorted = None

    def sorted_live(self):
        cached = self._sorted
        if cached is not None and cached[0] == self.generation:
            return cached[1]
        items = sorted(self.live.items())
        self._sorted = (self.generation, items)
        return items

    def clear_row_cache(self):
        self._rows.clear()
        self._row_bytes = 0

    def row_values(self, entry):
        cached = self._rows.get(entry)
        if cached is not None:
            return cached
        payload = self.log.read_at(entry[0], entry[1])
        row = values.unpack_row(self.columns, payload[17:])
        size = len(payload)
        if size <= ROW_CACHE_BUDGET:
            if self._row_bytes + size > ROW_CACHE_BUDGET:
                self.clear_row_cache()
            self._rows[entry] = row
            self._row_bytes += size
        return row

    def payload_of(self, entry):
        return self.log.read_at(entry[0], entry[1])

    def close(self):
        self.log.close()


class Transaction:
    """One explicit transaction, or the implicit transaction of one statement."""

    def __init__(self, ident, explicit):
        self.id = ident
        self.explicit = explicit
        self.overlay = {}   # table name -> {pk: entry or None}
        self.touched = []   # table names, first write order
        self._undo = []

    def table_overlay(self, name):
        overlay = self.overlay.get(name)
        if overlay is None:
            overlay = {}
            self.overlay[name] = overlay
            self.touched.append(name)
        return overlay

    def mark(self):
        return len(self._undo)

    def record_undo(self, table_name, pk, previous):
        self._undo.append((table_name, pk, previous))

    def rollback_to(self, mark):
        while len(self._undo) > mark:
            table_name, pk, previous = self._undo.pop()
            overlay = self.overlay[table_name]
            if previous is _ABSENT:
                overlay.pop(pk, None)
            else:
                overlay[pk] = previous

    def clear(self):
        self.overlay.clear()
        self.touched.clear()
        self._undo.clear()


class Engine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.catalog = storage.Catalog(data_dir)
        self.commit_log = storage.FrameLog(
            os.path.join(data_dir, "commit.log"), create=True
        )
        # The commit log has to be read first: a row is only visible when the
        # transaction id it carries reached the commit point.
        self.committed, self.max_txn = self._read_commit_log()
        self.tables = {}
        for name, meta in self.catalog.tables.items():
            self.tables[name] = Table(meta, data_dir, self.committed)
        self.txn = None
        self.next_txn = self.max_txn + 1
        self._purge_orphans()

    # --------------------------------------------------------------- recovery
    def _read_commit_log(self):
        committed = set()
        max_txn = 0
        valid = self.commit_log.valid_end()
        if valid != self.commit_log.size:
            self.commit_log.truncate(valid)
        for _offset, payload in self.commit_log.iter_frames():
            txn = storage._TXN.unpack_from(payload, 1)[0]
            max_txn = max(max_txn, txn)
            if payload[0] == storage.OP_COMMIT:
                committed.add(txn)
        return committed, max_txn

    def _has_orphans(self, table):
        for _offset, payload in table.log.iter_frames():
            txn = storage._TXN.unpack_from(payload, 1)[0]
            if txn != storage.BULK_TXN and txn not in self.committed:
                return True
        return False

    def _purge_orphans(self):
        """Rewrite tables that still carry frames of uncommitted transactions.

        Until this finishes, transaction ids could collide with stale orphans on
        the next start, so recovery does it before accepting a single request.
        """
        for table in list(self.tables.values()):
            if self._has_orphans(table):
                self._compact(table)

    # --------------------------------------------------------------- physical
    def _compact(self, table):
        rows = []
        for pk in sorted(table.live):
            payload = table.payload_of(table.live[pk])
            rows.append((pk, payload[17:]))
        table.live = table.log.rewrite(rows)
        table.clear_row_cache()
        table.bump()

    # -------------------------------------------------------------------- DDL
    def create_table(self, statement):
        name = statement["name"]
        if name in self.tables:
            raise errors.table_exists(name)
        columns = []
        primary = None
        for definition in statement["columns"]:
            column = {"name": definition["name"], "type": definition["type"]}
            if definition["type"] == "varchar":
                column["length"] = definition["length"]
            if definition["type"] == "vector":
                column["dim"] = definition["dim"]
            if definition["primary"]:
                if primary is not None:
                    raise errors.syntax("only one primary key is allowed")
                primary = definition["name"]
            columns.append(column)
        if statement["primary"]:
            if primary is not None and primary != statement["primary"]:
                raise errors.syntax("conflicting primary key definitions")
            primary = statement["primary"]
        if primary is None:
            raise errors.unsupported("a BIGINT primary key is required")
        seen = set()
        vector_columns = 0
        for column in columns:
            if column["name"] in seen:
                raise errors.syntax("duplicate column %s" % column["name"])
            seen.add(column["name"])
            if column["type"] == "vector":
                vector_columns += 1
        if vector_columns > 1:
            raise errors.unsupported("at most one vector column per table")
        primary_index = next((i for i, c in enumerate(columns) if c["name"] == primary), None)
        if primary_index is None:
            raise errors.syntax("primary key column is not defined")
        if columns[primary_index]["type"] != "bigint":
            raise errors.syntax("the primary key must be BIGINT")
        table_id = self.catalog.next_table_id
        meta = {
            "name": name,
            "id": table_id,
            "file": "table_%d.dat" % table_id,
            "columns": columns,
            "primary_index": primary_index,
            "indexes": [],
        }
        self.catalog.tables[name] = meta
        self.catalog.next_table_id = table_id + 1
        table = Table(meta, self.data_dir, self.committed, create=True)
        self.catalog.save()
        self.tables[name] = table
        return {"ok": True}

    def create_vector_index(self, statement):
        table = self.table(statement["table"])
        column_index = self.column_index(table, statement["column"])
        if table.columns[column_index]["type"] != "vector":
            raise errors.invalid_value("vector index needs a VECTOR column")
        for existing in table.meta["indexes"]:
            if existing["name"] == statement["name"]:
                raise errors.DBError("index_exists", "index already exists")
        table.meta["indexes"].append({"name": statement["name"], "column": column_index})
        self.catalog.save()
        return {"ok": True}

    # --------------------------------------------------------------- dispatch
    def execute(self, sql, mode="exact"):
        statement = sqlparser.parse(sql)
        kind = statement["kind"]
        if kind == "begin":
            if self.txn is not None:
                raise errors.unsupported("nested transactions are not supported")
            self.txn = Transaction(self._allocate_txn(), explicit=True)
            return {"ok": True}
        if kind in ("commit", "rollback"):
            if self.txn is None:
                return {"ok": True}
            if kind == "commit":
                self._commit(self.txn)
            else:
                self._abort(self.txn)
            return {"ok": True}
        if kind in ("create_table", "create_index"):
            if self.txn is not None:
                # DDL implicitly commits, matching MySQL, so a later ROLLBACK
                # cannot pretend the schema change never happened.
                self._commit(self.txn)
            if kind == "create_table":
                return self.create_table(statement)
            return self.create_vector_index(statement)
        implicit = self.txn is None
        transaction = self.txn or Transaction(self._allocate_txn(), explicit=False)
        mark = transaction.mark()
        try:
            result = self._run(transaction, statement, mode)
        except Exception:
            transaction.rollback_to(mark)
            self._refresh(transaction)
            if implicit:
                transaction.clear()
            raise
        if implicit:
            self._commit(transaction)
        return result

    def _run(self, transaction, statement, mode):
        kind = statement["kind"]
        if kind == "insert":
            return self._insert(transaction, statement)
        if kind == "select":
            return self._select(transaction, statement, mode)
        if kind == "update":
            return self._update(transaction, statement)
        if kind == "delete":
            return self._delete(transaction, statement)
        raise errors.unsupported("statement %s" % kind)

    def _allocate_txn(self):
        ident = self.next_txn
        self.next_txn += 1
        return ident

    def _refresh(self, transaction):
        for name in transaction.overlay:
            table = self.tables[name]
            table.clear_row_cache()
            table.bump()

    # ----------------------------------------------------------------- commit
    def _commit(self, transaction):
        for name in transaction.touched:
            self.tables[name].log.flush(sync=True)
        payload = storage.control_payload(storage.OP_COMMIT, transaction.id)
        self.commit_log.append([payload])
        self.commit_log.flush(sync=True)
        self.committed.add(transaction.id)
        self.max_txn = max(self.max_txn, transaction.id)
        for name in transaction.touched:
            table = self.tables[name]
            for pk, entry in transaction.overlay[name].items():
                if entry is None:
                    table.live.pop(pk, None)
                else:
                    table.live[pk] = entry
            table.bump()
        transaction.clear()
        if self.txn is transaction:
            self.txn = None

    def _abort(self, transaction):
        self._refresh(transaction)
        transaction.clear()
        if self.txn is transaction:
            self.txn = None

    # ---------------------------------------------------------------- helpers
    def table(self, name):
        table = self.tables.get(name)
        if table is None:
            raise errors.unknown_table(name)
        return table

    def column_index(self, table, name):
        for index, column in enumerate(table.columns):
            if column["name"] == name:
                return index
        raise errors.unknown_column(name)

    def lookup(self, transaction, table, pk):
        overlay = transaction.overlay.get(table.name)
        if overlay is not None and pk in overlay:
            return overlay[pk]
        return table.live.get(pk)

    def visible(self, transaction, table):
        """Sorted ``(pk, entry)`` pairs visible to the transaction."""
        overlay = transaction.overlay.get(table.name)
        if not overlay:
            return table.sorted_live()
        merged = dict(table.live)
        merged.update(overlay)
        return sorted((pk, entry) for pk, entry in merged.items() if entry is not None)

    def _append(self, transaction, table, op, pk, row_bytes=b""):
        self._append_many(transaction, table, op, [(pk, row_bytes)])

    def _append_many(self, transaction, table, op, rows):
        """Append many records of one statement with a single file write."""
        payloads = [
            storage.record_payload(op, transaction.id, pk, row_bytes)
            for pk, row_bytes in rows
        ]
        offsets = table.log.append(payloads)
        overlay = transaction.table_overlay(table.name)
        for (pk, _row_bytes), offset, payload in zip(rows, offsets, payloads):
            previous = overlay.get(pk, _ABSENT)
            transaction.record_undo(table.name, pk, previous)
            overlay[pk] = None if op == storage.OP_DELETE else (offset, len(payload))
        table.bump()

    # ------------------------------------------------------------ value rules
    def coerce(self, column, raw, what="value"):
        kind = column["type"]
        if kind == "bigint":
            return values.check_bigint(raw, what)
        if kind == "varchar":
            return values.check_varchar(raw, column["length"], what)
        if kind == "vector":
            return values.parse_vector_text(raw, column["dim"], what)
        raise errors.invalid_value("unknown column type")

    def literal(self, expr):
        if expr[0] == "lit":
            return expr[1]
        if expr[0] == "col":
            raise errors.invalid_value("column %s is not a literal" % expr[1])
        raise errors.invalid_value("only literals are allowed here")

    # ------------------------------------------------------------------- DML
    def _insert(self, transaction, statement):
        table = self.table(statement["table"])
        width = len(table.columns)
        targets = list(range(width))
        if statement["columns"] is not None:
            targets = [self.column_index(table, name) for name in statement["columns"]]
            if len(set(targets)) != width:
                raise errors.invalid_value("every column needs a value")
        prepared = []
        seen = set()
        for raw_row in statement["rows"]:
            if len(raw_row) != len(targets):
                raise errors.invalid_value("column count does not match value count")
            row = [None] * width
            for position, expr in enumerate(raw_row):
                column_position = targets[position]
                column = table.columns[column_position]
                row[column_position] = self.coerce(
                    column, self.literal(expr), column["name"]
                )
            pk = row[table.primary_index]
            if pk in seen:
                raise errors.duplicate_key("duplicate primary key %d in statement" % pk)
            seen.add(pk)
            if self.lookup(transaction, table, pk) is not None:
                raise errors.duplicate_key("duplicate primary key %d" % pk)
            prepared.append(row)
        self._append_many(
            transaction,
            table,
            storage.OP_UPSERT,
            [
                (row[table.primary_index], values.pack_row(table.columns, row))
                for row in prepared
            ],
        )
        return {"ok": True, "affected_rows": len(prepared)}

    def _predicates(self, table, where):
        prepared = []
        for left, operator, right in where or ():
            if left[0] == "col" and right[0] == "lit":
                column_position = self.column_index(table, left[1])
                operand = right[1]
            elif left[0] == "lit" and right[0] == "col":
                column_position = self.column_index(table, right[1])
                operand = left[1]
                operator = FLIPPED[operator]
            else:
                raise errors.unsupported("comparisons need one column and one literal")
            column = table.columns[column_position]
            if column["type"] == "vector":
                raise errors.unsupported("vector columns cannot be compared directly")
            prepared.append((column_position, operator, self.coerce(column, operand, column["name"])))
        return prepared

    def _matches(self, table, prepared, row):
        for column_position, operator, operand in prepared:
            if not COMPARATORS[operator](row[column_position], operand):
                return False
        return True

    def _update(self, transaction, statement):
        table = self.table(statement["table"])
        assignments = []
        for name, expr in statement["assign"]:
            column_position = self.column_index(table, name)
            if column_position == table.primary_index:
                raise errors.unsupported("updating the primary key is not supported")
            column = table.columns[column_position]
            assignments.append(
                (column_position, self.coerce(column, self.literal(expr), column["name"]))
            )
        prepared = self._predicates(table, statement["where"])
        affected = 0
        for pk, entry in self.visible(transaction, table):
            row = list(table.row_values(entry))
            if prepared and not self._matches(table, prepared, row):
                continue
            for column_position, value in assignments:
                row[column_position] = value
            self._append(
                transaction, table, storage.OP_UPSERT, pk, values.pack_row(table.columns, row)
            )
            affected += 1
        return {"ok": True, "affected_rows": affected}

    def _delete(self, transaction, statement):
        table = self.table(statement["table"])
        prepared = self._predicates(table, statement["where"])
        affected = 0
        for pk, entry in self.visible(transaction, table):
            if prepared:
                row = table.row_values(entry)
                if not self._matches(table, prepared, row):
                    continue
            self._append(transaction, table, storage.OP_DELETE, pk)
            affected += 1
        return {"ok": True, "affected_rows": affected}

    # ---------------------------------------------------------------- SELECT
    def _select(self, transaction, statement, mode):
        table = self.table(statement["table"])
        items = self._prepare_items(table, statement["items"])
        predicates = self._predicates(table, statement["where"])
        order = self._prepare_order(table, statement["order"], items)
        limit = statement["limit"]
        entries = self.visible(transaction, table)
        if not DISABLE_COLUMNAR and _np is not None:
            try:
                snapshot = self._snapshot(transaction, table, entries)
                return self._select_columnar(
                    table, items, predicates, order, limit, snapshot
                )
            except _FallbackToRows:
                pass
        return self._select_rows(table, entries, items, predicates, order, limit)

    def _prepare_items(self, table, items):
        prepared = []
        names = []
        for item in items:
            expression = item["expr"]
            if expression[0] == "star":
                for index, column in enumerate(table.columns):
                    prepared.append(("col", index))
                    names.append(column["name"])
                continue
            resolved = self._prepare_expr(table, expression, allow_l2=True)
            prepared.append(resolved)
            if item["alias"]:
                names.append(item["alias"])
            elif resolved[0] == "col":
                names.append(table.columns[resolved[1]]["name"])
            elif resolved[0] == "l2":
                names.append("distance")
            else:
                names.append("column_%d" % len(names))
        return {"exprs": prepared, "names": names}

    def _prepare_expr(self, table, expr, allow_l2):
        kind = expr[0]
        if kind == "col":
            return ("col", self.column_index(table, expr[1]))
        if kind == "lit":
            return ("lit", expr[1])
        if kind == "l2":
            if not allow_l2:
                raise errors.syntax("L2_DISTANCE is not allowed here")
            sides = []
            for side in (expr[1], expr[2]):
                if side[0] == "col":
                    position = self.column_index(table, side[1])
                    if table.columns[position]["type"] != "vector":
                        raise errors.invalid_value("L2_DISTANCE needs a VECTOR column")
                    sides.append(("col", position))
                elif side[0] == "lit":
                    sides.append(("vec", self._literal_vector(table, side[1])))
                else:
                    raise errors.syntax("unsupported L2_DISTANCE argument")
            if sides[0][0] == "vec" and sides[1][0] == "vec":
                raise errors.invalid_value("L2_DISTANCE needs at least one vector column")
            return ("l2", sides[0], sides[1])
        raise errors.syntax("unsupported expression")

    def _literal_vector(self, table, raw):
        if table.vector_index is None:
            raise errors.invalid_value("table has no VECTOR column")
        return values.parse_vector_text(
            raw, table.columns[table.vector_index]["dim"], "vector literal"
        )

    def _prepare_order(self, table, order, items):
        prepared = []
        for item in order:
            expression = item["expr"]
            key = None
            if expression[0] == "col":
                for index, name in enumerate(items["names"]):
                    if name == expression[1]:
                        key = ("item", index)
                        break
                if key is None:
                    key = ("col", self.column_index(table, expression[1]))
            else:
                key = self._prepare_expr(table, expression, allow_l2=True)
            prepared.append((key, item["desc"]))
        return prepared

    # ------------------------------------------------------- columnar fast path
    def _snapshot(self, transaction, table, entries):
        # The generation counter is bumped by every write, commit and rollback,
        # so it is enough to keep the snapshot in sync with visible rows.
        cached = table._snapshot
        if cached is not None and cached[0] == table.generation:
            return cached[1]
        estimated = len(entries) * (
            sum(self._column_width(column) for column in table.columns) + 48
        )
        if estimated > COLUMNAR_BUDGET:
            table._snapshot = None
            raise _FallbackToRows()
        count = len(entries)
        pks = _np.empty(count, dtype=_np.int64)
        vectors = None
        if table.vector_index is not None:
            vectors = _np.empty(
                (count, table.columns[table.vector_index]["dim"]), dtype=_np.float32
            )
        # Pre-create one entry per column: an empty table must still answer a
        # projection instead of raising a KeyError on the missing array.
        numbers = {
            position: []
            for position, column in enumerate(table.columns)
            if column["type"] == "bigint"
        }
        strings = {
            position: []
            for position, column in enumerate(table.columns)
            if column["type"] == "varchar"
        }
        for index in range(count):
            pk, entry = entries[index]
            row = table.row_values(entry)
            pks[index] = pk
            for position, column in enumerate(table.columns):
                if column["type"] == "vector":
                    vectors[index] = row[position]
                elif column["type"] == "bigint":
                    numbers.setdefault(position, []).append(row[position])
                else:
                    strings.setdefault(position, []).append(row[position])
        snapshot = {
            "count": count,
            "pks": pks,
            "vectors": vectors,
            "numbers": {key: _np.array(value, dtype=_np.int64) for key, value in numbers.items()},
            "strings": strings,
        }
        # Dictionary-encode VARCHAR columns: equality filters then compare int32
        # arrays instead of running a Python comparison per row.
        codes = {}
        code_of = {}
        for position, column_values in strings.items():
            mapping = {}
            vocabulary = []
            encoded = _np.empty(len(column_values), dtype=_np.int32)
            for index, text in enumerate(column_values):
                code = mapping.get(text)
                if code is None:
                    code = len(vocabulary)
                    mapping[text] = code
                    vocabulary.append(text)
                encoded[index] = code
            codes[position] = encoded
            code_of[position] = mapping
        snapshot["codes"] = codes
        snapshot["code_of"] = code_of
        table._snapshot = (table.generation, snapshot)
        return snapshot

    def _column_width(self, column):
        if column["type"] == "bigint":
            return 8
        if column["type"] == "vector":
            return 4 * column["dim"]
        return 4

    def _select_columnar(self, table, items, predicates, order, limit, snapshot):
        indices = self._filtered_indices(table, snapshot, predicates)
        pks = snapshot["pks"]
        if not order:
            selected = indices[:limit] if limit is not None else indices
        else:
            keys = []
            for key, descending in order:
                key_values = self._order_values(table, snapshot, items, key, indices)
                if descending and key_values.dtype == object:
                    raise _FallbackToRows()
                keys.append((key_values, descending))
            if len(keys) == 1:
                selected = _top_k(keys[0][0], pks[indices], keys[0][1], limit)
                selected = indices[selected]
            else:
                lex_keys = [pks[indices]]
                for key_values, descending in reversed(keys):
                    lex_keys.append(-key_values if descending else key_values)
                ordering = _np.lexsort(lex_keys)
                selected = indices[ordering]
                if limit is not None:
                    selected = selected[:limit]
        return {
            "ok": True,
            "columns": items["names"],
            "rows": self._emit_columnar(table, snapshot, items, selected),
        }

    def _filtered_indices(self, table, snapshot, predicates):
        count = snapshot["count"]
        mask = None
        for column_position, operator, operand in predicates:
            column = table.columns[column_position]
            if column["type"] == "bigint":
                data = snapshot["numbers"][column_position]
                comparison = {
                    "=": data == operand,
                    "<>": data != operand,
                    "<": data < operand,
                    "<=": data <= operand,
                    ">": data > operand,
                    ">=": data >= operand,
                }[operator]
            elif operator in ("=", "<>") and count:
                code = snapshot["code_of"][column_position].get(operand)
                data = snapshot["codes"][column_position]
                if code is None:
                    comparison = _np.zeros(count, dtype=bool)
                else:
                    comparison = data == code
                if operator == "<>":
                    comparison = _np.logical_not(comparison)
            else:
                data = snapshot["strings"][column_position]
                comparison = _np.fromiter(
                    (COMPARATORS[operator](value, operand) for value in data),
                    dtype=bool,
                    count=count,
                )
            mask = comparison if mask is None else _np.logical_and(mask, comparison)
        if mask is None:
            return _np.arange(count)
        return _np.nonzero(mask)[0]

    def _order_values(self, table, snapshot, items, key, indices):
        kind = key[0]
        if kind == "item":
            expression = items["exprs"][key[1]]
            if expression[0] == "l2":
                return self._distance_column(table, snapshot, expression, indices)
            if expression[0] == "col":
                return self._column_values(table, snapshot, expression[1], indices)
            raise errors.unsupported("cannot order by a literal")
        if kind == "col":
            return self._column_values(table, snapshot, key[1], indices)
        if kind == "l2":
            return self._distance_column(table, snapshot, key, indices)
        raise errors.unsupported("cannot order by a literal")

    def _column_values(self, table, snapshot, column_position, indices):
        column = table.columns[column_position]
        if column["type"] == "bigint":
            return snapshot["numbers"][column_position][indices]
        if column["type"] == "varchar":
            data = snapshot["strings"][column_position]
            return _np.array([data[i] for i in indices], dtype=object)
        raise errors.unsupported("cannot order by a VECTOR column")

    def _distance_column(self, table, snapshot, expression, indices):
        vectors = snapshot["vectors"][indices]
        left, right = expression[1], expression[2]
        if right[0] == "vec":
            query = _np.array(right[1], dtype=_np.float32)
        elif left[0] == "vec":
            query = _np.array(left[1], dtype=_np.float32)
        else:
            raise errors.unsupported("L2_DISTANCE needs one vector column")
        delta = vectors - query
        squared = _np.einsum("ij,ij->i", delta, delta, dtype=_np.float32)
        return _np.sqrt(squared)

    def _emit_columnar(self, table, snapshot, items, selected):
        buffers = []
        for expression in items["exprs"]:
            if expression[0] == "col":
                column = table.columns[expression[1]]
                if column["type"] == "vector":
                    buffers.append(("matrix", snapshot["vectors"][selected]))
                else:
                    buffers.append(
                        ("array", self._column_values(table, snapshot, expression[1], selected))
                    )
            elif expression[0] == "lit":
                buffers.append(("const", [expression[1]] * len(selected)))
            elif expression[0] == "l2":
                buffers.append(("array", self._distance_column(table, snapshot, expression, selected)))
            else:  # pragma: no cover
                raise errors.unsupported("unsupported select item")
        rows = []
        for position in range(len(selected)):
            row = []
            for kind, buffer in buffers:
                if kind == "matrix":
                    row.append([float(element) for element in buffer[position]])
                    continue
                value = buffer[position]
                if isinstance(value, _np.generic):
                    value = value.item()
                row.append(value)
            rows.append(row)
        return rows

    # ------------------------------------------------------------ row by row
    def _select_rows(self, table, entries, items, predicates, order, limit):
        rows = []
        for pk, entry in entries:
            row = table.row_values(entry)
            if predicates and not self._matches(table, predicates, row):
                continue
            rows.append((pk, row))
        if order:
            def sort_key(pair):
                pk, row = pair
                key = []
                for expression, descending in order:
                    value = self._eval_row(table, items, row, expression)
                    key.append(_Descending(value) if descending else value)
                key.append(pk)
                return tuple(key)

            rows.sort(key=sort_key)
        if limit is not None:
            rows = rows[:limit]
        output = []
        for _pk, row in rows:
            output.append(
                [self._eval_row(table, items, row, expression) for expression in items["exprs"]]
            )
        return {"ok": True, "columns": items["names"], "rows": output}

    def _eval_row(self, table, items, row, expression):
        kind = expression[0]
        if kind == "col":
            return row[expression[1]]
        if kind == "lit":
            return expression[1]
        if kind == "item":
            return self._eval_row(table, items, row, items["exprs"][expression[1]])
        if kind == "l2":
            left, right = expression[1], expression[2]
            left_value = row[left[1]] if left[0] == "col" else left[1]
            right_value = row[right[1]] if right[0] == "col" else right[1]
            return values.vector_distance(left_value, right_value)
        raise errors.unsupported("unsupported expression")

    # --------------------------------------------------------------- shutdown
    def shutdown(self):
        if self.txn is not None:
            self._abort(self.txn)
        for table in self.tables.values():
            self._compact(table)
        # Every table now holds only bulk rows, so the commit log can be reset.
        self.commit_log.truncate(storage.HEADER)
        self.commit_log.flush(sync=True)
        for table in self.tables.values():
            table.log.flush(sync=True)
            table.close()


class _Descending:
    """Wrapper that flips comparison order inside a sort key."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return other.value < self.value

    def __eq__(self, other):
        return other.value == self.value


def _top_k(key_values, pks, descending, limit):
    """Top-k by one key with the primary key as tie breaker.

    Selecting every row that reaches the k-th value before cutting avoids
    dropping tied rows in favour of a larger primary key.
    """
    ordered = -key_values if descending else key_values
    count = len(ordered)
    if limit is None or limit >= count:
        return _np.lexsort((pks, ordered))
    if limit <= 0:
        return _np.empty(0, dtype=_np.int64)
    threshold = _np.partition(ordered, limit - 1)[limit - 1]
    keep = _np.nonzero(ordered <= threshold)[0]
    order = _np.lexsort((pks[keep], ordered[keep]))
    return keep[order][:limit]
