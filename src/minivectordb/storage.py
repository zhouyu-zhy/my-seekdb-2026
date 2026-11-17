"""Crash safe storage: framed append-only log per table plus a JSON catalog.

Frame layout (little endian)::

    +----------+----------+---------------------------+
    | crc32(4) | len(4)   | payload(len)              |
    +----------+----------+---------------------------+

    payload = op(1) | txn(8) | pk(8) | row bytes        # op 1 upsert, 2 delete
    payload = op(1) | txn(8)                            # op 3 commit, 4 abort

A record is only visible after its transaction's commit frame is durable, so a
torn tail can always be dropped without inventing half transactions.
Transaction id 0 marks rows written by compaction: they are already committed by
construction because the rewritten file is fsynced and atomically renamed.
"""

import os
import struct
import zlib

from . import errors

MAGIC = b"MVD1\n"
HEADER = len(MAGIC)
_CRC = struct.Struct("<I")
_LEN = struct.Struct("<I")
_TXN = struct.Struct("<q")
_PK = struct.Struct("<q")

OP_UPSERT = 1
OP_DELETE = 2
OP_COMMIT = 3
OP_ABORT = 4

BULK_TXN = 0


def frame_bytes(payload):
    return _CRC.pack(zlib.crc32(payload) & 0xFFFFFFFF) + _LEN.pack(len(payload)) + payload


def record_payload(op, txn, pk, data=b""):
    return bytes((op,)) + _TXN.pack(txn) + _PK.pack(pk) + data


def control_payload(op, txn):
    return bytes((op,)) + _TXN.pack(txn)


def payload_size(row_bytes):
    return 17 + len(row_bytes)


class FrameLog:
    """One append-only file. Only the writer ever appends at the tail."""

    def __init__(self, path, create=False):
        self.path = path
        exists = os.path.exists(path)
        if not exists and not create:
            raise errors.DBError("storage_error", "missing table file %s" % path)
        self._fh = open(path, "r+b" if exists else "w+b")
        if not exists:
            self._fh.write(MAGIC)
            self._fh.flush()
            os.fsync(self._fh.fileno())
        self.size = os.fstat(self._fh.fileno()).st_size
        if self.size < HEADER:
            self._fh.seek(0)
            self._fh.write(MAGIC)
            self._fh.truncate(HEADER)
            self.size = HEADER

    # ------------------------------------------------------------------- read
    def read_at(self, offset, length):
        self._fh.seek(offset)
        data = self._fh.read(length)
        if len(data) != length:
            raise errors.DBError("storage_error", "short read at %d" % offset)
        return data

    def iter_frames(self, start=HEADER):
        position = start
        size = self.size
        while position + 8 <= size:
            header = self.read_at(position, 8)
            checksum = _CRC.unpack(header[:4])[0]
            length = _LEN.unpack(header[4:])[0]
            if length < 9 or position + 8 + length > size:
                return
            payload = self.read_at(position + 8, length)
            if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
                return
            yield position + 8, payload
            position += 8 + length

    def valid_end(self):
        """Offset of the first byte that is not part of a verified frame."""
        end = HEADER
        for offset, payload in self.iter_frames():
            end = offset + len(payload)
        return end

    # ------------------------------------------------------------------ write
    def append(self, payloads):
        """Write frames at the tail and return each payload offset."""
        if not payloads:
            return []
        start = self.size
        chunks = []
        offsets = []
        position = start
        for payload in payloads:
            frame = frame_bytes(payload)
            offsets.append(position + 8)
            chunks.append(frame)
            position += len(frame)
        self._fh.seek(start)
        self._fh.write(b"".join(chunks))
        self.size = position
        return offsets

    def flush(self, sync=True):
        if self._fh.closed:
            return
        self._fh.flush()
        if sync:
            try:
                os.fsync(self._fh.fileno())
            except OSError:  # pragma: no cover - platform dependent
                pass

    def truncate(self, position):
        self._fh.flush()
        self._fh.truncate(position)
        self.size = position

    # ----------------------------------------------------------------- rewrite
    def rewrite(self, rows):
        """Replace the file with one committed row per entry.

        ``rows`` is an iterable of ``(pk, row_bytes)``. The new file is fsynced
        and renamed over the old one, so a crash keeps either the old or the new
        version, never a mixture.
        """
        temp_path = self.path + ".compact"
        offsets = {}
        with open(temp_path, "wb") as temp:
            temp.write(MAGIC)
            position = HEADER
            buffer = []
            buffered = 0
            for pk, row_bytes in rows:
                payload = record_payload(OP_UPSERT, BULK_TXN, pk, row_bytes)
                frame = frame_bytes(payload)
                offsets[pk] = (position + 8, len(payload))
                buffer.append(frame)
                buffered += len(frame)
                position += len(frame)
                if buffered >= (1 << 20):
                    temp.write(b"".join(buffer))
                    buffer = []
                    buffered = 0
            if buffer:
                temp.write(b"".join(buffer))
            temp.flush()
            os.fsync(temp.fileno())
        self._fh.close()
        os.replace(temp_path, self.path)
        self._fh = open(self.path, "r+b")
        self.size = position
        _sync_directory(os.path.dirname(self.path) or ".")
        return offsets

    def close(self):
        if not self._fh.closed:
            self._fh.close()


def _sync_directory(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - Windows has no directory fsync
        pass
    finally:
        os.close(fd)


def recover(path, committed, create=False):
    """Open a table file and rebuild ``{pk: (offset, length)}``.

    ``committed`` is the set of transaction ids that reached the commit record in
    ``commit.log``.  Rows of any other transaction are ignored: they belong to a
    transaction that was rolled back or killed before the commit point.
    """
    log = FrameLog(path, create=create)
    valid = log.valid_end()
    if valid != log.size:
        log.truncate(valid)
    live = {}
    for offset, payload in log.iter_frames():
        op = payload[0]
        txn = _TXN.unpack_from(payload, 1)[0]
        if op in (OP_COMMIT, OP_ABORT):
            continue
        if txn != BULK_TXN and txn not in committed:
            continue
        pk = _PK.unpack_from(payload, 9)[0]
        if op == OP_DELETE:
            live.pop(pk, None)
        else:
            live[pk] = (offset, len(payload))
    return log, live


class Catalog:
    """Small JSON file describing every table and index."""

    def __init__(self, data_dir):
        self.path = os.path.join(data_dir, "catalog.json")
        self.tables = {}
        self.next_table_id = 1
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        import json

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (ValueError, OSError):
            # A truncated catalog can only happen if the process died between
            # writing and renaming; the previous copy is still in place.
            raise errors.DBError("storage_error", "catalog is unreadable")
        self.tables = document.get("tables", {})
        self.next_table_id = document.get("next_table_id", 1)

    def save(self):
        import json

        document = {"format": 1, "next_table_id": self.next_table_id, "tables": self.tables}
        temp_path = self.path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)
        _sync_directory(os.path.dirname(self.path) or ".")
