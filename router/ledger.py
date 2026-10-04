"""`tamias-router ledger-info`: the shape of a SQLite file, and nothing else.

This exists so the Tamias Observer ledger's tables and columns can be seen
before anything knows what they mean. For each table it prints the table name,
its columns with their declared types, and how many rows the table holds.

It never selects a row's contents. It never interprets a column name, infers a
unit, or claims what a value means: this module reports structure only, and
what it cannot see it does not describe. The database is opened read-only, so
inspecting a ledger cannot change it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .readonly import ReadOnlyError, open_read_only

#: Printed first. The reason this command exists, and the limit of it.
LEDGER_NOTE = "Ledger schema only (read-only). No row contents are read or printed."

#: SQLite's own bookkeeping tables. Not the ledger's, and not worth printing.
_INTERNAL_PREFIX = "sqlite_"


@dataclass(frozen=True)
class LedgerColumn:
    """One column: its name and the type the schema declares."""

    name: str
    declared_type: str


@dataclass(frozen=True)
class LedgerTable:
    """One table: its name, its columns, and how many rows it holds."""

    name: str
    columns: tuple[LedgerColumn, ...]
    rows: int


def inspect_ledger(path: Path | str) -> tuple[LedgerTable, ...]:
    """Every user table in `path`, with columns and row counts. Read-only."""
    with open_read_only(path) as connection:
        names = _table_names(connection)
        return tuple(
            LedgerTable(
                name=name,
                columns=_columns(connection, name),
                rows=_row_count(connection, name),
            )
            for name in names
        )


def _table_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    return tuple(str(row[0]) for row in rows if not str(row[0]).startswith(_INTERNAL_PREFIX))


def _columns(connection: sqlite3.Connection, table: str) -> tuple[LedgerColumn, ...]:
    # The table-valued PRAGMA takes the table name as a bound value, so no
    # identifier ever has to be built out of a name read from the file.
    rows = connection.execute(
        "SELECT name, type FROM pragma_table_info(?)", (table,)
    ).fetchall()
    return tuple(LedgerColumn(name=str(row[0]), declared_type=str(row[1] or "")) for row in rows)


def _row_count(connection: sqlite3.Connection, table: str) -> int:
    # COUNT(*) reads no column value, so no row content can reach the output.
    row = connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()
    return int(row[0])


def quote_identifier(name: str) -> str:
    """A SQL identifier quoted for use as one."""
    return '"' + str(name).replace('"', '""') + '"'


def format_ledger(path: Path | str, tables: tuple[LedgerTable, ...]) -> str:
    """Render the schema. Table names, column names, declared types, counts."""
    lines = [LEDGER_NOTE, f"ledger: {Path(path)}"]
    if not tables:
        return "\n".join([*lines, "tables: none"])

    lines.append(f"tables: {len(tables)}")
    for table in tables:
        lines.append("")
        lines.append(table.name)
        lines.append(f"  {'rows':<{_LABEL_WIDTH}} {table.rows}")
        lines.append("  columns")
        if table.columns:
            width = max(len(column.name) for column in table.columns)
            lines.extend(
                f"    {column.name:<{width}}  {column.declared_type or _NO_TYPE}"
                for column in table.columns
            )
        else:
            lines.append(f"    {_NO_TYPE}")
    return "\n".join(lines)


#: Shown where the schema declares no type for a column.
_NO_TYPE = "(no declared type)"

_LABEL_WIDTH = 8


__all__ = [
    "LEDGER_NOTE",
    "LedgerColumn",
    "LedgerTable",
    "ReadOnlyError",
    "format_ledger",
    "inspect_ledger",
    "quote_identifier",
]