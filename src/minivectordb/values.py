"""Value level rules: BIGINT, VARCHAR(n) and VECTOR(n) as float32."""

import math
import struct

from . import errors

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1

_PACK_F32 = struct.Struct("<f")
_UNPACK_F32 = struct.Struct("<f").unpack
_PACK_I64 = struct.Struct("<q")
_PACK_U32 = struct.Struct("<I")


def to_float32(value, what="value"):
    """Round a Python float to the nearest float32 and reject non-finite results.

    The database stores vectors as float32, so every value that enters storage
    and every query vector go through this function.  ``struct.pack`` raises
    ``OverflowError`` when the double is outside the float32 range, which is the
    conversion overflow the task asks us to reject.
    """
    try:
        packed = _PACK_F32.pack(value)
    except OverflowError:
        raise errors.invalid_vector("%s is not representable as float32" % what)
    result = _UNPACK_F32(packed)[0]
    if not math.isfinite(result):
        raise errors.invalid_vector("%s must be a finite float32" % what)
    return result


def parse_vector_text(text, dim, what="vector"):
    """Parse a '[...]' literal into a list of float32 values of length ``dim``."""
    if not isinstance(text, str):
        raise errors.invalid_vector("%s must be a '[...]' literal" % what)
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        raise errors.invalid_vector("%s must be a '[...]' literal" % what)
    body = body[1:-1].strip()
    parts = [] if not body else [chunk.strip() for chunk in body.split(",")]
    if len(parts) != dim:
        raise errors.dimension_mismatch(
            "%s has %d dimension(s), expected %d" % (what, len(parts), dim)
        )
    values = []
    for index, chunk in enumerate(parts):
        if not chunk:
            raise errors.invalid_vector("%s has an empty element" % what)
        try:
            number = float(chunk)
        except ValueError:
            raise errors.invalid_vector("%s has a non numeric element: %r" % (what, chunk))
        if not math.isfinite(number):
            raise errors.invalid_vector("%s element %d is not finite" % (what, index))
        values.append(to_float32(number, "%s element %d" % (what, index)))
    return values


def check_bigint(value, what="value"):
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.invalid_value("%s must be a BIGINT" % what)
    if value < INT64_MIN or value > INT64_MAX:
        raise errors.invalid_value("%s is out of BIGINT range" % what)
    return value


def check_varchar(value, length, what="value"):
    if not isinstance(value, str):
        raise errors.invalid_value("%s must be a string" % what)
    if len(value) > length:
        raise errors.value_too_long(
            "%s has %d characters, limit is %d" % (what, len(value), length)
        )
    return value


def pack_row(columns, values):
    """Encode one row with the physical layout of its table."""
    out = bytearray()
    for column, value in zip(columns, values):
        kind = column["type"]
        if kind == "bigint":
            out += _PACK_I64.pack(value)
        elif kind == "varchar":
            raw = value.encode("utf-8")
            out += _PACK_U32.pack(len(raw))
            out += raw
        elif kind == "vector":
            out += struct.pack("<%df" % len(value), *value)
        else:  # pragma: no cover - catalog is validated on load
            raise errors.invalid_value("unknown column type %r" % kind)
    return bytes(out)


def unpack_row(columns, payload):
    """Decode one row; returns a tuple in column declaration order."""
    values = []
    offset = 0
    for column in columns:
        kind = column["type"]
        if kind == "bigint":
            values.append(_PACK_I64.unpack_from(payload, offset)[0])
            offset += 8
        elif kind == "varchar":
            (length,) = _PACK_U32.unpack_from(payload, offset)
            offset += 4
            values.append(payload[offset:offset + length].decode("utf-8"))
            offset += length
        elif kind == "vector":
            dim = column["dim"]
            values.append(list(struct.unpack_from("<%df" % dim, payload, offset)))
            offset += 4 * dim
        else:  # pragma: no cover
            raise errors.invalid_value("unknown column type %r" % kind)
    return tuple(values)


def vector_distance(left, right):
    """Euclidean distance between two float32 lists (double accumulation)."""
    total = 0.0
    for a, b in zip(left, right):
        diff = a - b
        total += diff * diff
    return math.sqrt(total)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)
