"""Error codes returned to the judge through the JSONL protocol."""


class DBError(Exception):
    """A statement level failure. The session stays alive after it."""

    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def duplicate_key(message="duplicate primary key"):
    return DBError("duplicate_key", message)


def dimension_mismatch(message="vector dimension mismatch"):
    return DBError("dimension_mismatch", message)


def invalid_vector(message="invalid vector value"):
    return DBError("invalid_vector", message)


def unsupported(message="unsupported statement"):
    return DBError("unsupported_sql", message)


def syntax(message="syntax error"):
    return DBError("syntax_error", message)


def unknown_table(name):
    return DBError("table_not_found", "unknown table: %s" % name)


def unknown_column(name):
    return DBError("unknown_column", "unknown column: %s" % name)


def table_exists(name):
    return DBError("table_exists", "table already exists: %s" % name)


def value_too_long(message="value exceeds declared length"):
    return DBError("value_too_long", message)


def invalid_value(message="invalid value"):
    return DBError("invalid_value", message)
