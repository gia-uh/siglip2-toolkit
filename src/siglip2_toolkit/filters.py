"""Serializable, general-purpose filters over mappings.

Filters are plain, JSON-serializable expressions over any dict-like subject
(typically a record's ``metadata``, but nothing ties them to images, EXIF
schemas or storage engines). Example::

    from siglip2_toolkit.filters import field, And, Not

    f = And(
        field("width").between(1000, 4000),
        field("camera.make").contains("nikon"),
        Not(field("tags").exists()),
    )
    matches({"width": 2000, "camera": {"make": "NIKON Z6"}}, f)  # -> True

A missing field never matches (any operator, including ``!=``); compose with
``Not``/``Or`` when you want the opposite. Order comparisons coerce numeric
strings (``"240.0"`` compares as a number) and fall back to lexicographic
comparison when both sides are strings. ``contains``/``startswith`` are
case-insensitive by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "And",
    "Field",
    "Filter",
    "Not",
    "Or",
    "field",
    "from_dict",
    "matches",
    "to_dict",
]

_MISSING = object()
_COMPARISONS = ("==", "!=", "<", "<=", ">", ">=")
_STRING_OPS = ("contains", "startswith")
_OPS = _COMPARISONS + ("in", "exists") + _STRING_OPS


@dataclass(frozen=True)
class Filter:
    """A leaf filter: ``path`` (dotted), ``op`` and ``value``."""

    path: str
    op: str
    value: Any = None
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unknown filter operator: {self.op!r}")


class Field:
    """Builder for filter leaves over a (possibly dotted) field path."""

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = str(path)

    # Comparison operators (return filters, not booleans).
    def __eq__(self, value: Any) -> Filter:  # type: ignore[override]
        return Filter(self.path, "==", value)

    def __ne__(self, value: Any) -> Filter:  # type: ignore[override]
        return Filter(self.path, "!=", value)

    def __lt__(self, value: Any) -> Filter:
        return Filter(self.path, "<", value)

    def __le__(self, value: Any) -> Filter:
        return Filter(self.path, "<=", value)

    def __gt__(self, value: Any) -> Filter:
        return Filter(self.path, ">", value)

    def __ge__(self, value: Any) -> Filter:
        return Filter(self.path, ">=", value)

    def between(self, low: Any, high: Any) -> "And":
        """Convenience: ``low <= field <= high``."""
        return And(self.__ge__(low), self.__le__(high))

    def in_(self, values: Iterable[Any]) -> Filter:
        """Membership: the field value must be one of ``values``."""
        return Filter(self.path, "in", list(values))

    def contains(self, value: str, case_sensitive: bool = False) -> Filter:
        """Substring test for string fields."""
        return Filter(self.path, "contains", value, case_sensitive=case_sensitive)

    def startswith(self, value: str, case_sensitive: bool = False) -> Filter:
        """Prefix test for string fields."""
        return Filter(self.path, "startswith", value, case_sensitive=case_sensitive)

    def exists(self) -> Filter:
        """The field is present and not empty (None/""/[]/{} count as empty)."""
        return Filter(self.path, "exists")

    def __repr__(self) -> str:
        return f"field({self.path!r})"


def field(path: str) -> Field:
    """Start a filter expression over ``path`` (dotted paths allowed)."""
    return Field(path)


@dataclass(frozen=True, init=False)
class And:
    """Every child filter must match."""

    filters: tuple


@dataclass(frozen=True, init=False)
class Or:
    """At least one child filter must match."""

    filters: tuple


@dataclass(frozen=True)
class Not:
    """Negates a child filter (missing fields included)."""

    filter: Any


def _make_combinator(cls):
    def __init__(self, *filters: Any) -> None:
        object.__setattr__(self, "filters", tuple(filters))

    cls.__init__ = __init__
    return cls


And = _make_combinator(And)
Or = _make_combinator(Or)


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #


def _resolve(subject: Mapping[str, Any], path: str) -> Any:
    current: Any = subject
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _is_empty(value: Any) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, (list, dict, tuple, set)) and len(value) == 0


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _compare(value: Any, expected: Any, op: str) -> bool:
    left_number, right_number = _as_number(value), _as_number(expected)
    if left_number is not None and right_number is not None:
        left, right = left_number, right_number
    else:
        left, right = value, expected

    if op == "==":
        return left == right
    if op == "!=":
        return left != right

    if not (isinstance(left, str) and isinstance(right, str)):
        if not (isinstance(left, float) and isinstance(right, float)):
            return False
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right  # ">="


def _matches_leaf(leaf: Filter, subject: Mapping[str, Any]) -> bool:
    value = _resolve(subject, leaf.path)
    if value is _MISSING:
        return False

    if leaf.op == "exists":
        return not _is_empty(value)

    if leaf.op == "in":
        try:
            return value in leaf.value
        except TypeError:
            return False

    if leaf.op in _STRING_OPS:
        if not isinstance(value, str):
            return False
        haystack, needle = value, str(leaf.value)
        if not leaf.case_sensitive:
            haystack, needle = haystack.lower(), needle.lower()
        return needle in haystack if leaf.op == "contains" else haystack.startswith(needle)

    return _compare(value, leaf.value, leaf.op)


def matches(subject: Mapping[str, Any], node: Any) -> bool:
    """Whether ``subject`` (a mapping) satisfies ``node``."""
    if isinstance(node, And):
        return all(matches(subject, child) for child in node.filters)
    if isinstance(node, Or):
        return any(matches(subject, child) for child in node.filters)
    if isinstance(node, Not):
        return not matches(subject, node.filter)
    if isinstance(node, Filter):
        return _matches_leaf(node, subject)
    raise TypeError(f"not a filter expression: {node!r}")


# --------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------- #


def to_dict(node: Any) -> dict[str, Any]:
    """Serialize a filter expression to plain JSON-compatible data."""
    if isinstance(node, And):
        return {"op": "and", "filters": [to_dict(child) for child in node.filters]}
    if isinstance(node, Or):
        return {"op": "or", "filters": [to_dict(child) for child in node.filters]}
    if isinstance(node, Not):
        return {"op": "not", "filter": to_dict(node.filter)}
    if isinstance(node, Filter):
        data: dict[str, Any] = {"path": node.path, "op": node.op}
        if node.op != "exists":
            data["value"] = node.value
        if node.case_sensitive:
            data["case_sensitive"] = True
        return data
    raise TypeError(f"not a filter expression: {node!r}")


def from_dict(data: Mapping[str, Any]) -> Any:
    """Rebuild a filter expression from :func:`to_dict` output."""
    op = data.get("op")
    if op in ("and", "or"):
        children = [from_dict(child) for child in data.get("filters", [])]
        return And(*children) if op == "and" else Or(*children)
    if op == "not":
        return Not(from_dict(data.get("filter", {})))
    if "path" in data:
        return Filter(
            str(data["path"]),
            str(op),
            data.get("value"),
            case_sensitive=bool(data.get("case_sensitive", False)),
        )
    raise ValueError(f"cannot rebuild filter from: {data!r}")
