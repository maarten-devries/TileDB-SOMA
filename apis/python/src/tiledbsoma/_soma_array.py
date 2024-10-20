# Copyright (c) 2021-2023 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2023 TileDB, Inc.
#
# Licensed under the MIT License.

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
from somacore import options
from typing_extensions import Self

from . import _tdb_handles, _util

# This package's pybind11 code
from . import pytiledbsoma as clib  # noqa: E402
from ._soma_object import SOMAObject
from ._types import OpenTimestamp, is_nonstringy_sequence
from .options._soma_tiledb_context import SOMATileDBContext


class SOMAArray(SOMAObject[_tdb_handles.SOMAArrayWrapper[Any]]):
    """Base class for all SOMAArrays: DataFrame and NDarray.

    Lifecycle:
        Maturing.
    """

    __slots__ = ()

    @classmethod
    def open(
        cls,
        uri: str,
        mode: options.OpenMode = "r",
        *,
        tiledb_timestamp: Optional[OpenTimestamp] = None,
        context: Optional[SOMATileDBContext] = None,
        platform_config: Optional[options.PlatformConfig] = None,
        clib_type: Optional[str] = None,
    ) -> Self:
        """Opens this specific type of SOMA object."""
        retval = super().open(
            uri,
            mode,
            tiledb_timestamp=tiledb_timestamp,
            context=context,
            platform_config=platform_config,
            clib_type="SOMAArray",
        )
        return retval

    @property
    def schema(self) -> pa.Schema:
        """Returns data schema, in the form of an
        `Arrow Schema <https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html>`_.

        Lifecycle:
            Maturing.
        """
        return self._clib_handle.schema

    @property
    def ndim(self) -> int:
        return len(self._clib_handle.dimension_names)

    def config_options_from_schema(self) -> clib.PlatformConfig:
        """Returns metadata about the array that is not encompassed within the
        Arrow Schema, in the form of a PlatformConfig.

        Available attributes are:
            * dataframe_dim_zstd_level: int
            * sparse_nd_array_dim_zstd_level: int
            * sparse_nd_array_dim_zstd_level: int
            * write_X_chunked: bool
            * goal_chunk_nnz: int
            * remote_cap_nbytes: int
            * capacity: int
            * offsets_filters: str
                * name (of filter): str
                * compression_level: str
            * validity_filters: str
            * attrs: str
                * name (of attribute): str
                    * filters: str
                        * name (of filter): str
                        * compression_level: str
            * dims: str
                * name (of dimension): str
                    * filters: str
                        * name (of filter): str
                        * compression_level: str
                    * tile: int
            * allows_duplicates: bool
            * tile_order: str
            * cell_order: str
            * consolidate_and_vacuum: bool
        """
        return self._clib_handle.config_options_from_schema()

    def _tiledb_array_keys(self) -> Tuple[str, ...]:
        """Return all dim and attr names."""
        return self._tiledb_dim_names() + self._tiledb_attr_names()

    def _tiledb_dim_names(self) -> Tuple[str, ...]:
        """Reads the dimension names from the schema: for example, ['obs_id', 'var_id']."""
        return tuple(self._clib_handle.dimension_names)

    def _tiledb_attr_names(self) -> Tuple[str, ...]:
        """Reads the attribute names from the schema:
        for example, the list of column names in a dataframe.
        """
        return self.attr_names

    @property
    def dim_names(self) -> Tuple[str, ...]:
        return tuple(self._clib_handle.dimension_names)

    @property
    def attr_names(self) -> Tuple[str, ...]:
        return tuple(
            f.name
            for f in self.schema
            if f.name not in self._clib_handle.dimension_names
        )

    def _cast_domainish(
        self, domainish: List[Any]
    ) -> Tuple[Tuple[object, object], ...]:
        result = []
        for i, slot in enumerate(domainish):

            arrow_type = slot[0].type
            if pa.types.is_timestamp(arrow_type):
                pandas_type = np.dtype(arrow_type.to_pandas_dtype())
                result.append(
                    tuple(
                        pandas_type.type(e.cast(pa.int64()).as_py(), arrow_type.unit)
                        for e in slot
                    )
                )
            else:
                result.append(tuple(e.as_py() for e in slot))

        return tuple(result)

    def non_empty_domain(self) -> Tuple[Tuple[Any, Any], ...]:
        """
        Retrieves the non-empty domain for each dimension, namely the smallest
        and largest indices in each dimension for which the array/dataframe has
        data occupied.  This is nominally the same as the domain used at
        creation time, but if for example only a portion of the available domain
        has actually had data written, this function will return a tighter
        range.
        """
        return self._cast_domainish(self._clib_handle.non_empty_domain())

    def _domain(self) -> Tuple[Tuple[Any, Any], ...]:
        """This is the SOMA domain, not the core domain.
        * For arrays with core current-domain support:
          o soma domain is core current domain
          o soma maxdomain is core domain
        * For arrays without core current-domain support:
          o soma domain is core domain
          o soma maxdomain is core domain
          o core current domain is not accessed at the soma level
        * Core domain has been around forever and is immutable.
        * Core current domain is new as of core 2.25 and can be
          resized up to core (max) domain.
        """
        return self._cast_domainish(self._clib_handle.domain())

    def _maxdomain(self) -> Tuple[Tuple[Any, Any], ...]:
        """This is the SOMA maxdomain, not the core domain.
        * For arrays with core current-domain support:
          o soma domain is core current domain
          o soma maxdomain is core domain
        * For arrays without core current-domain support:
          o soma domain is core domain
          o soma maxdomain is core domain
          o core current domain is not accessed at the soma level
        * Core domain has been around forever and is immutable.
        * Core current domain is new as of core 2.25 and can be
          resized up to core (max) domain.
        """
        return self._cast_domainish(self._clib_handle.maxdomain())

    def _set_reader_coords(self, sr: clib.SOMAArray, coords: Sequence[object]) -> None:
        """Parses the given coords and sets them on the SOMA Reader."""
        if not is_nonstringy_sequence(coords):
            raise TypeError(
                f"coords type {type(coords)} must be a regular sequence,"
                " not str or bytes"
            )

        if len(coords) > self.ndim:
            raise ValueError(
                f"coords ({len(coords)} elements) must be shorter than ndim"
                f" ({self.ndim})"
            )
        for i, coord in enumerate(coords):
            dim = self.schema.field(i)
            if not self._set_reader_coord(sr, i, dim, coord):
                raise TypeError(
                    f"coord type {type(coord)} for dimension {dim.name}"
                    f" (slot {i}) unsupported"
                )

    def _set_reader_coord(
        self, sr: clib.SOMAArray, dim_idx: int, dim: pa.Field, coord: object
    ) -> bool:
        """Parses a single coordinate entry.

        The base implementation parses the most fundamental types shared by all
        TileDB Array types; subclasses can implement their own readers that
        handle types not recognized here.

        Returns:
            True if successful, False if unrecognized.
        """
        if coord is None:
            return True  # No constraint; select all in this dimension

        if isinstance(coord, int):
            sr.set_dim_points_int64(dim.name, [coord])
            return True
        if isinstance(coord, slice):
            _util.validate_slice(coord)
            try:
                dom = self._domain()[dim_idx]
                lo_hi = _util.slice_to_numeric_range(coord, dom)
            except _util.NonNumericDimensionError:
                return False  # We only handle numeric dimensions here.
            if lo_hi:
                sr.set_dim_ranges_int64(dim.name, [lo_hi])
            # If `None`, coord was `slice(None)` and there is no constraint.
            return True
        return False
