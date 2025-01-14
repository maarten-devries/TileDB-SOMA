"""Utilities for use with hypothesis -- mostly search strategies."""

from __future__ import annotations

import datetime
from itertools import pairwise
from typing import Any, Mapping, Sequence

import hypothesis.extra.numpy as ht_np
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from hypothesis import strategies as st

from tests.ht._ht_test_config import HT_TEST_CONFIG

Shape = tuple[int, ...]
ArrowSlice = tuple[int, int]


def everything_except(excluded_types: type) -> st.SearchStrategy[type]:
    """Create a strategy for all types exclusive of those specified.

    Example:
        everything_except(int|float)

    """
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def resolve_dtype(
    draw: st.DrawFn,
    dtype: npt.DTypeLike | pa.DataType | st.SearchStrategy[npt.DTypeLike | pa.DataType],
) -> np.dtype:
    """resolve the dtype argument to a numpy.dtype. Helper for search strategies."""
    if isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)
    if isinstance(dtype, pa.DataType):
        dtype = dtype.to_pandas_dtype()
    dtype = np.dtype(dtype)
    return dtype


def from_datatype(
    datatype: pa.DataType, *args, **kwargs
) -> st.SearchStrategy[pa.Scalar]:
    """Strategy to return an element of the given type."""
    if datatype in [pa.binary(), pa.large_binary()]:
        return st.binary(*args, **kwargs).map(lambda v: pa.scalar(v, type=datatype))
    elif datatype in [pa.string(), pa.large_string()]:
        return st.text(*args, **kwargs).map(lambda v: pa.scalar(v, type=datatype))
    elif datatype == pa.null():
        return st.none()
    elif pa.types.is_timestamp(datatype):
        allow_nan = kwargs.get("allow_nan", False)

        # NEP-7 defines the NaT value as integer -2**63
        min_value = pa.scalar(kwargs.get("min_value", -(2**63) + 1), type=datatype)
        max_value = pa.scalar(kwargs.get("max_value", 2**63 - 1), type=datatype)

        elems = (
            st.integers(min_value.value, max_value.value) | st.none()
            if allow_nan
            else st.integers(min_value.value, max_value.value)
        )
        return elems.map(lambda v: pa.scalar(v, type=datatype))
    else:
        return ht_np.from_dtype(
            np.dtype(datatype.to_pandas_dtype()), *args, **kwargs
        ).map(lambda v: pa.scalar(v, type=datatype))


def tiledb_timestamps(from_future: bool = False):
    """Strategy which generates POSIX / TileDB timestamps, aka ints from 0 to now.

    NB: bug sc-61054 is triggered with timestamp==0, so generate only timestamps 0<ts<=now.
    """
    # TODO: the min_value should be zero, but this triggers sc-61054 in 1.15.0. Update when fixed.
    BEGINNING_OF_TIME = 1 if HT_TEST_CONFIG["sc-61054_workaround"] else 0

    # technically TileDB allows timestamps up to 2**64-1, but Python time_t can't represent it, so
    # settle for something less (not as far in the future).
    END_OF_TIME = (
        datetime.datetime.now()
        if not from_future
        else datetime.datetime.fromtimestamp(2**37)
    )

    return st.one_of(
        st.none(),
        st.integers(
            min_value=BEGINNING_OF_TIME,
            max_value=int(END_OF_TIME.timestamp()),
        ),
        st.datetimes(
            min_value=datetime.datetime.fromtimestamp(BEGINNING_OF_TIME),
            max_value=END_OF_TIME,
        ),
    )


@st.composite
def arrow_integer_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(st.sampled_from((pa.int8(), pa.int16(), pa.int32(), pa.int64())))


@st.composite
def arrow_unsigned_integer_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(st.sampled_from((pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64())))


@st.composite
def arrow_floating_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(st.sampled_from((pa.float16(), pa.float32(), pa.float64())))


@st.composite
def arrow_timestamp_datatypes(draw: st.DrawFn) -> pa.DataType:
    return pa.timestamp(
        unit=draw(st.sampled_from(("s", "ms", "us", "ns"))),
        tz=draw(st.sampled_from((None, "UTC", "Europe/London"))),
    )


@st.composite
def arrow_datetime_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(
        st.sampled_from(
            (
                pa.time32(unit=draw(st.sampled_from(("s", "ms")))),
                pa.time64(unit=draw(st.sampled_from(("us", "ns")))),
                pa.date32(),
                pa.date64(),
            )
        )
    )


@st.composite
def arrow_decimal_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(
        st.sampled_from(
            (
                pa.decimal128(
                    precision=draw(st.integers(min_value=1, max_value=38)),
                    scale=draw(st.integers(min_value=-(2**31), max_value=2**31 - 1)),
                ),
                pa.decimal256(
                    precision=draw(st.integers(min_value=1, max_value=76)),
                    scale=draw(st.integers(min_value=-(2**31), max_value=2**31 - 1)),
                ),
            )
        )
    )


@st.composite
def arrow_datatypes(draw: st.DrawFn) -> pa.DataType:
    return draw(
        st.one_of(
            arrow_integer_datatypes(),
            arrow_unsigned_integer_datatypes(),
            arrow_floating_datatypes(),
            st.sampled_from(
                (
                    pa.null(),
                    pa.bool_(),
                    pa.binary(length=draw(st.integers(min_value=-1, max_value=1024))),
                    pa.string(),
                    pa.large_binary(),
                    pa.large_string(),
                )
            ),
            arrow_timestamp_datatypes(),
            arrow_datetime_datatypes(),
            arrow_decimal_datatypes(),
        )
    )


def ndarray_datatype() -> st.SearchStrategy:
    return st.from_type(pa.DataType).filter(
        lambda t: (
            pa.types.is_primitive(t)
            and not (pa.types.is_timestamp(t) and t.tz is not None)
            and not pa.types.is_time(t)
            and not pa.types.is_date(t)
            and t
            not in [
                pa.float16(),
            ]
        )
    )


@st.composite
def arrow_schema_field_name(draw: st.DrawFn) -> str:
    # TileDB attribute names may not start with '__'
    elements = st.text(min_size=1).filter(lambda n: not n.startswith("__"))
    if HT_TEST_CONFIG["sc-61291_workaround"]:
        elements = elements.filter(lambda n: "\x00" not in n)
    return draw(elements)


@st.composite
def arrow_schema(
    draw: st.DrawFn,
    max_fields: int | None = None,
    unique_field_names: bool = False,
    required_fields: Sequence[pa.Field] = (),
    elements: st.SearchStrategy[pa.DataType] | None = None,
) -> pa.Schema:
    # A schema must have at least one index column and one attribute column
    max_fields = 100 if max_fields is None else max_fields
    assert max_fields > 1

    fields = {f.name: f for f in required_fields}
    if "soma_joinid" not in fields and draw(st.booleans()):
        fields["soma_joinid"] = pa.field("soma_joinid", nullable=False, type=pa.int64())

    elements = arrow_datatypes() if elements is None else elements

    # NB: no metadata in Arrow schema
    n_fields = draw(st.integers(min_value=2, max_value=max_fields))
    for n in range(n_fields - len(fields)):
        while True:
            field_name = draw(arrow_schema_field_name())
            if not unique_field_names or field_name not in fields:
                break

        field_type = draw(elements)
        field_nullable = True if field_type == pa.null() else draw(st.booleans())
        fields[field_name] = pa.field(
            field_name, nullable=field_nullable, type=field_type
        )

    return pa.schema(list(fields.values()))


@st.composite
def arrow_shape(
    draw: st.DrawFn,
    shape: (
        int | st.SearchStrategy[int] | Shape | st.SearchStrategy[Shape] | None
    ) = None,
) -> Shape:
    if isinstance(shape, st.SearchStrategy):
        shape = draw(shape)
    if shape is None:
        shape = draw(st.integers(max_value=1024))
    if isinstance(shape, np.generic):
        shape = shape.item()
    if isinstance(shape, np.ndarray):
        shape = tuple(shape.tolist())
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, tuple) and len(shape) == 1 and shape[0] >= 0:
        return shape
    raise ValueError("Invalid shape argument - specify 1D shape.")


@st.composite
def arrow_slice(draw: st.DrawFn, size: int) -> ArrowSlice:
    """Return (offset, length) suitable for Array.slice or ChunkedArray.slice."""
    if size <= 0:
        return (0, 0)
    offset = draw(st.integers(min_value=0, max_value=size - 1))
    length = draw(st.integers(min_value=0, max_value=size - offset - 1))
    return (offset, length)


def pad_array(arr: npt.NDArray[Any], draw: st.DrawFn) -> pa.Array:
    """Strategy helper: add padding to one or both ends of the array. This tests for Arrow array "offset" handling."""

    if HT_TEST_CONFIG.get("sc-61239_workaround", False):
        return pa.array(arr)

    head = draw(st.integers(min_value=0, max_value=16))
    tail = draw(st.integers(min_value=0, max_value=16))
    if not bool(head or tail):
        return pa.array(arr)

    padding = draw(st.from_type(arr.dtype.type))

    shape = (arr.shape[0] + head + tail, *arr.shape[1:])
    padded_arr = np.empty_like(arr, shape=shape)
    padded_arr[0:head] = padding
    padded_arr[head : head + len(arr)] = arr
    padded_arr[head + len(arr) :] = padding
    return pa.array(padded_arr)[head : head + len(arr)]


@st.composite
def arrow_array(
    draw: st.DrawFn,
    dtype: npt.DTypeLike | pa.DataType | st.SearchStrategy[npt.DTypeLike | pa.DataType],
    shape: int | st.SearchStrategy[int] | Shape | st.SearchStrategy[Shape],
    *,
    elements: st.SearchStrategy[Any] | Mapping[str, Any] | None = None,
    fill: st.SearchStrategy[Any] | None = None,
    unique: bool = False,
    padding: bool = True,
) -> pa.Array:
    """Wrapper around hypothesis.extra.numpy.arrays, which returns value as a PyArrow Array.

    NB: this is quite slow for large arrays. See arrow_array_fast for a limited, but faster variant.
    This variant retained for flexibility (vs speed).
    """
    dtype = resolve_dtype(draw, dtype)
    shape = draw(arrow_shape(shape))

    if (
        not HT_TEST_CONFIG["allow_nullable"]
        and dtype.kind in ["m", "M"]
        and elements is None
    ):
        # NaT gets turned into a nulled position by pyarrow.array
        elements = {"allow_nan": False}

    nparr = draw(
        ht_np.arrays(
            dtype=dtype, shape=shape, unique=unique, elements=elements, fill=fill
        )
    )
    arr = pad_array(nparr, draw) if padding else pa.array(nparr)

    # sanity check
    assert HT_TEST_CONFIG["allow_nullable"] or not pa.compute.any(arr.is_null()).as_py()

    return arr


@st.composite
def arrow_array_fast(
    draw: st.DrawFn,
    dtype: npt.DTypeLike | pa.DataType | st.SearchStrategy[npt.DTypeLike | pa.DataType],
    shape: int | st.SearchStrategy[int] | Shape | st.SearchStrategy[Shape],
    *,
    unique: bool = False,
    padding: bool = True,
    min_value: Any = None,
    max_value: Any = None,
) -> pa.Array:
    """Faster, but limited version of arrow_array search strategy.

    Only supports a subset of types, all positions randomly generated (no fill),
    and no control over element values (e.g., no ``elements`` argument). Importantly
    this means no NaN for floats, etc.
    """

    def gen_unique_floats(
        rng: np.random.Generator, lo: float, hi: float, n: int
    ) -> npt.NDArray[np.float64]:
        out = np.empty(n)
        needed = n
        while needed != 0:
            arr = rng.uniform(lo, hi, needed)
            uniqs = np.setdiff1d(np.unique(arr), out[: n - needed])
            out[n - needed : n - needed + uniqs.size] = uniqs
            needed -= uniqs.size
        rng.shuffle(out)
        return out

    dtype = resolve_dtype(draw, dtype)

    shape = draw(arrow_shape(shape))
    length = shape[0]

    rng = np.random.default_rng(seed=draw(st.integers(min_value=0)))
    match dtype.kind:
        case "f":
            low = min_value if min_value is not None else -np.finfo(dtype).max / 2
            high = max_value if max_value is not None else np.finfo(dtype).max / 2
            if unique:
                nparr = gen_unique_floats(rng, low, high, length).astype(dtype)
            else:
                nparr = rng.uniform(low, high=high, size=length).astype(dtype)

        case "i" | "u":
            # RNG draws max of int64
            low = int(min_value) if min_value is not None else -np.iinfo(dtype).max
            high = int(max_value) if max_value is not None else np.iinfo(dtype).max
            if (high - low) < np.iinfo(np.int64).max:
                if high > low:
                    nparr = rng.choice(high - low, size=length, replace=(not unique))
                else:
                    nparr = np.full(shape=shape, fill_value=low, dtype=dtype)
                nparr += low
            else:
                nparr = rng.choice(
                    np.iinfo(np.int64).max, size=length, replace=(not unique)
                )
                if min_value is not None:
                    nparr += low
                else:
                    nparr -= np.iinfo(dtype).max // 2

            nparr = nparr.astype(dtype)

        case "M":
            # TODO: implement min_value/max_value
            assert min_value is None and max_value is None
            nparr = rng.choice(
                np.iinfo(np.int64).max, size=length, replace=(not unique)
            )
            nparr = nparr.astype(dtype)

        case "b":
            # TODO: implement min_value/max_value
            assert min_value is None and max_value is None
            nparr = rng.choice([True, False], size=length, replace=(not unique))

        case _:
            raise TypeError(f"Unsupported dtype: {dtype}")

    return pad_array(nparr, draw) if padding else pa.array(nparr)


@st.composite
def arrow_chunked_array_fast(
    draw: st.DrawFn,
    dtype: npt.DTypeLike | pa.DataType | st.SearchStrategy[npt.DTypeLike | pa.DataType],
    shape: int | st.SearchStrategy[int] | Shape | st.SearchStrategy[Shape],
    *,
    unique: bool = False,
    padding: bool = True,
    min_value: Any = None,
    max_value: Any = None,
    splits: int | Sequence[int] | st.SearchStrategy[Sequence[int]] | None = None,
) -> pa.ChunkedArray:

    shape = draw(arrow_shape(shape))
    length = shape[0]

    arr = draw(
        arrow_array_fast(
            dtype=dtype,
            shape=length,
            unique=unique,
            padding=padding,
            min_value=min_value,
            max_value=max_value,
        )
    )

    # sometimes, we want multiple (separate) underlying arrays, just to mix things up and
    # ensure we are not generating only contiguous buffers.
    if draw(st.booleans()):
        # split in half, and copy values of second half to entirely different memory location.
        # Must round-trip through NumPy as PyArrow doesn't appear to have a "copy" operator
        # available (except in very recent versions of the package).
        #
        first_half = arr[0 : len(arr) // 2]
        second_half = arr[len(arr) // 2 :].to_numpy().copy()
        if padding:
            second_half = pad_array(second_half, draw)
        arr = pa.chunked_array([first_half, second_half])

    if splits is None:
        splits = 0
    elif isinstance(splits, st.SearchStrategy):
        splits = draw(splits)
    if isinstance(splits, int):
        splits = draw(splitss(n_splits=min(splits, len(arr)), max_value=len(arr)))

    return split_arrow_array(arr, splits)


@st.composite
def splitss(
    draw: st.DrawFn, n_splits: int | st.SearchStrategy[int], max_value: int
) -> list[int]:
    if n_splits == 0:
        return []
    rng = np.random.default_rng(seed=draw(st.integers(min_value=0)))
    splits = rng.choice(max_value, size=n_splits, replace=False)
    splits.sort()
    return splits.tolist()


def split_arrow_array(
    arr: pa.Array | pa.ChunkedArray, splits: list[int]
) -> pa.ChunkedArray:
    assert np.array_equal(np.unique(splits), splits), "splits not unique and sorted"
    assert len(splits) == 0 or (
        splits[0] >= 0 and splits[-1] < len(arr)
    ), "splits out of range"

    split_points = [0] + splits + [len(arr)]
    arr_splits = [arr[st:sp] for st, sp in pairwise(split_points)]
    return pa.chunked_array(arr_splits, type=arr.type)


@st.composite
def random_length_tuple(
    draw, elements=st.integers(), min_length: int = 0, max_length: int = 10
):
    """Generates a tuple of random length with elements drawn from the provided strategy."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return tuple(draw(st.lists(elements, min_size=length, max_size=length)))


@st.composite
def contiguous_slices(draw: Any, size: int) -> slice:
    """Generates slices that will select indices up to the supplied size, always with
    stride of 1.

    Based on hypothesis.strategies.slices()
    """
    if size == 0:
        step = draw(st.sampled_from([None, 1]))
        return slice(None, None, step)

    # For slices start is inclusive and stop is exclusive
    start = draw(st.integers(0, size) | st.none())
    stop = draw(st.integers(0, size) | st.none())
    start, stop = (stop, start) if (start or 0) > (stop or 0) else (start, stop)
    step = 1

    if draw(st.booleans()) and start is not None:
        start -= size
    if draw(st.booleans()) and stop is not None:
        stop -= size
    if (not draw(st.booleans())) and step == 1:
        step = None

    return slice(start, stop + 1 if stop is not None else stop, step)


def schemas_equal(s1: pa.Schema, s2: pa.Schema, ignore_field_order=False) -> bool:
    """NB: assumes all field names are unique! Raises if not."""
    if not ignore_field_order:
        return s1 == s2
    else:
        if len(s2) != len(s1):
            return False
        return all(
            s1.field(field_name) == s2.field(field_name) for field_name in s1.names
        )


def arrays_equal(
    read: pa.ChunkedArray, expected: pa.ChunkedArray, *, equal_nan: bool = False
) -> bool:
    """Zero copy test for array equality, optionally allowing NaN==NaN."""

    # TODO: handle nullable arrays

    if (read.type != expected.type) or (len(read) != len(expected)):
        return False

    if not pa.types.is_floating(expected.type):
        return expected.equals(read)

    # Floating point path, to allow for NaN. Implemented with NumPy for convenience only
    return all(
        np.array_equal(r.to_numpy(), e.to_numpy(), equal_nan=equal_nan)
        for r, e in zip(read.chunks, expected.chunks)
    )


def tables_equal(
    read: pa.Table, expected: pa.Table, *, equal_nan: bool = False
) -> bool:
    """Test for table equality, optionally allowing NaN==NaN."""

    # TODO: handle nullable arrays

    read_schema = read.schema
    expected_schema = expected.schema

    # checking field order and length up front simplifies code below
    if [f.name for f in read_schema] != [f.name for f in expected_schema]:
        return False

    if HT_TEST_CONFIG["sc-61222_workaround"]:
        # because sc-61222, where read returns tables with nullable missing from
        # the schema, we need to cast the table before comparing.
        read_schema = pa.schema([f.with_nullable(False) for f in read_schema])
        expected_schema = pa.schema([f.with_nullable(False) for f in expected_schema])

    if HT_TEST_CONFIG["sc-61227_workaround"]:
        # because sc-61227, read returns `int64` when we expect `timestamp[us]`
        for fidx, field in enumerate(expected.schema):
            if (
                field.type == pa.timestamp("us")
                and read_schema.field(fidx).type == pa.int64()
            ):
                read_schema = read_schema.set(
                    fidx, read_schema.field(fidx).with_type(field.type)
                )

    if (read_schema != expected_schema) or len(read) != len(expected):
        return False

    expected = expected.cast(expected_schema)
    read = read.cast(read_schema)
    is_eq = all(arrays_equal(r, e, equal_nan=equal_nan) for r, e in zip(read, expected))
    return is_eq


def df_to_table(df: pd.DataFrame, *, schema: pa.Schema | None = None) -> pa.Table:

    # Table.from_pandas attempts to infer nulled values (e.g., NaN->null, NaT->null).
    # We often do not want this behavior, so explicitly override it with `from_pandas=False`
    # paramter of pa.array().

    # NB: this doesn't work with NaT/timestamp64. We could pass a `mask` param to `to_numpy`,
    # but NaT is such a strange beast, leaving as is for now.

    schema = pa.Schema.from_pandas(df).remove_metadata() if schema is None else schema
    tbl = pa.Table.from_pydict(
        {k: pa.array(v, from_pandas=False) for k, v in df.items()}, schema=schema
    )
    return tbl


def posix_filename() -> st.SearchStrategy:
    return st.text(
        alphabet=st.characters(
            codec="ascii",
            categories=["Lu", "Ll", "Nd"],
            include_characters=["_", "-", "."],
        ),
        min_size=1,
    ).filter(lambda fn: fn not in [".", ".."])
