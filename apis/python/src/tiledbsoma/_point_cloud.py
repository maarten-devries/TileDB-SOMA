# Copyright (c) 2021-2023 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2023 TileDB, Inc.
#
# Licensed under the MIT License.

"""
Implementation of a SOMA Point Cloud
"""

from typing import Any, Final, Mapping, Optional, Self, Sequence, Tuple, Union

import pyarrow as pa
import somacore
from somacore import coordinates, options

from . import _arrow_types, _util
from . import pytiledbsoma as clib
from ._exception import SOMAError, map_exception_for_create
from ._query_condition import QueryCondition
from ._read_iters import TableReadIter
from ._soma_array import SOMAArray
from ._tdb_handles import PointCloudWrapper
from ._types import OpenTimestamp
from .options._soma_tiledb_context import _validate_soma_tiledb_context
from .options._tiledb_create_write_options import (
    TileDBCreateOptions,
    TileDBWriteOptions,
)


class PointCloud(SOMAArray, somacore.SpatialDataFrame):
    """A multi-column table with spatial indexing and a user-defined schema.
    Lifecycle: experimental
    """

    _wrapper_type = PointCloudWrapper

    __slots__ = ("_axis_names",)
    soma_type: Final = "SOMAPointCloud"  # type: ignore[misc]

    @classmethod
    def create(
        cls,
        uri: str,
        *,
        schema: pa.Schema,
        index_column_names: Sequence[str] = (options.SOMA_JOINID, "x", "y"),
        axis_names: Sequence[str] = ("x", "y"),
        domain: Optional[Sequence[Optional[Tuple[Any, Any]]]] = None,
        platform_config: Optional[options.PlatformConfig] = None,
        context: Optional[Any] = None,
        tiledb_timestamp: Optional[OpenTimestamp] = None,
    ):
        # ) -> Self:
        """Creates a new ``PointCloud`` at the given URI.

        The schema of the created point cloud  will include a column named
        ``soma_joinid`` of type ``pyarrow.int64``, with negative values disallowed, and
        at least one axis with numeric type.  If a ``soma_joinid`` column is
        present in the provided schema, it must be of the correct type.  If the
        ``soma_joinid`` column is not provided, one will be added. The ``soma_joinid``
        may be an index column. The axis columns must be index columns.

        Args:
            uri: The URI where the dataframe will be created.
            schema: Arrow schema defining the per-column schema. This schema
                must define all columns, including columns to be named as index
                columns.  If the schema includes types unsupported by the SOMA
                implementation, an error will be raised.
            index_column_names: A list of column names to use as user-defined index
                columns (e.g., ``['x', 'y']``). All named columns must exist in the
                schema, and at least one index column name is required.
            axis_names: An ordered list of axis column names that correspond to the
                names of axes of the the coordinate space the points are defined on.
                Must be the name of index columns.
            domain: An optional sequence of tuples specifying the domain of each
                index column. Each tuple should be a pair consisting of the minimum
                and maximum values storable in the index column. If omitted entirely,
                or if ``None`` in a given dimension, the corresponding index-column
                domain will use the minimum and maximum possible values for the
                column's datatype.  This makes a point cloud dataframe growable.

        Returns:
            The newly created geometry dataframe, opened for writing.

        Lifecycle: experimental
        """
        context = _validate_soma_tiledb_context(context)
        schema = _util.canonicalize_schema(schema, index_column_names)

        if not set(axis_names).issubset(index_column_names):
            raise ValueError(
                f"Axis names {axis_names} must be defined in index columns ({index_column_names})"
            )

        cls._axis_names = axis_names

        if domain is None:
            domain = tuple(None for _ in index_column_names)
        else:
            ndom = len(domain)
            nidx = len(index_column_names)
            if ndom != nidx:
                raise ValueError(
                    f"if domain is specified, it must have the same length as index_column_names; got {ndom} != {nidx}"
                )

        index_column_schema = []
        index_column_data = {}

        for index_column_name, slot_soma_domain in zip(index_column_names, domain):
            pa_field = schema.field(index_column_name)
            dtype = _arrow_types.tiledb_type_from_arrow_type(
                pa_field.type, is_indexed_column=True
            )

            (slot_core_current_domain, saturated_cd) = _util.fill_out_slot_soma_domain(
                slot_soma_domain, index_column_name, pa_field.type, dtype
            )
            (slot_core_max_domain, saturated_md) = _util.fill_out_slot_soma_domain(
                None, index_column_name, pa_field.type, dtype
            )

            extent = _util.find_extent_for_domain(
                index_column_name,
                TileDBCreateOptions.from_platform_config(platform_config),
                dtype,
                slot_core_current_domain,
            )

            # Necessary to avoid core array-creation error "Reduce domain max by
            # 1 tile extent to allow for expansion."
            slot_core_current_domain = _util.revise_domain_for_extent(
                slot_core_current_domain, extent, saturated_cd
            )
            slot_core_max_domain = _util.revise_domain_for_extent(
                slot_core_max_domain, extent, saturated_md
            )

            # Here is our Arrow data API for communicating schema info between
            # Python/R and C++ libtiledbsoma:
            #
            # [0] core max domain lo
            # [1] core max domain hi
            # [2] core extent parameter

            index_column_schema.append(pa_field)

            index_column_data[pa_field.name] = [*slot_core_current_domain, extent]

        index_column_info = pa.RecordBatch.from_pydict(
            index_column_data, schema=pa.schema(index_column_schema)
        )

        plt_cfg = _util.build_clib_platform_config(platform_config)
        timestamp_ms = context._open_timestamp_ms(tiledb_timestamp)
        try:
            clib.SOMAPointCloud.create(
                uri,
                schema=schema,
                index_column_info=index_column_info,
                ctx=context.native_context,
                platform_config=plt_cfg,
                timestamp=(0, timestamp_ms),
            )
        except SOMAError as e:
            raise map_exception_for_create(e, uri) from None

        handle = cls._wrapper_type.open(uri, "w", context, tiledb_timestamp)
        return cls(
            handle,
            _dont_call_this_use_create_or_open_instead="tiledbsoma-internal-code",
        )

    def read(
        self,
        coords: options.SparseDFCoords = (),
        column_names: Optional[Sequence[str]] = None,
        *,
        batch_size: options.BatchSize = options.BatchSize(),
        partitions: Optional[options.ReadPartitions] = None,
        result_order: options.ResultOrderStr = options.ResultOrder.AUTO,
        value_filter: Optional[str] = None,
        platform_config: Optional[options.PlatformConfig] = None,
        # ) -> data.ReadIter[pa.Table]:
    ):
        """Reads a user-defined slice of data into Arrow tables.

        Args:
            coords: for each index dimension, which rows to read.
                Defaults to ``()``, meaning no constraint -- all IDs.
            column_names: the named columns to read and return.
                Defaults to ``None``, meaning no constraint -- all column names.
            partitions: If present, specifies that this is part of
                a partitioned read, and which part of the data to include.
            result_order: the order to return results, specified as a
                :class:`~options.ResultOrder` or its string value.
            value_filter: an optional value filter to apply to the results.
                The default of ``None`` represents no filter. Value filter
                syntax is implementation-defined; see the documentation
                for the particular SOMA implementation for details.
        Returns:
            A :class:`ReadIter` of :class:`pa.Table`s.

        Lifecycle: experimental
        """
        del batch_size  # Currently unused.
        _util.check_unpartitioned(partitions)
        self._check_open_read()

        handle = self._handle._handle

        context = handle.context()
        if platform_config is not None:
            config = context.tiledb_config.copy()
            config.update(platform_config)
            context = clib.SOMAContext(config)

        sr = clib.SOMADataFrame.open(
            uri=handle.uri,
            mode=clib.OpenMode.read,
            context=context,
            column_names=column_names or [],
            result_order=_util.to_clib_result_order(result_order),
            timestamp=handle.timestamp and (0, handle.timestamp),
        )

        if value_filter is not None:
            sr.set_condition(QueryCondition(value_filter), handle.schema)

        self._set_reader_coords(sr, coords)

        # # TODO: batch_size
        return TableReadIter(sr)

    def read_region(
        self,
        region: Optional[options.SpatialRegion] = None,
        column_names: Optional[Sequence[str]] = None,
        *,
        extra_coords: Optional[Mapping[str, options.SparseDFCoord]] = None,
        transform: Optional[coordinates.CoordinateTransform] = None,
        region_coord_space: Optional[coordinates.CoordinateSpace] = None,
        batch_size: options.BatchSize = options.BatchSize(),
        partitions: Optional[options.ReadPartitions] = None,
        result_order: options.ResultOrderStr = options.ResultOrder.AUTO,
        value_filter: Optional[str] = None,
        platform_config: Optional[options.PlatformConfig] = None,
        # ) -> "SpatialRead[data.ReadIter[pa.Table]]":
    ):
        """Reads a data intersecting an user-defined region into a
        :class:`SpatialRead` with data in Arrow tables.


        Args:
            region: The region to query. May be a box in the form
                [x_min, y_min, x_max, y_max] (for 2D images), a box in the form
                [x_min, y_min, z_min, x_max, y_max, z_max] (for 3D images), or
                a shapely Geometry.
            column_names: the named columns to read and return.
                Defaults to ``None``, meaning no constraint -- all column names.
            extra_coords: a name to coordinate mapping non-spatial index columns.
                Defaults to selecting entire region for non-spatial coordinates.
            transform: An optional coordinate transform that provides desribes the
                Defaults to ``None``, meaning an identity transform.
            region_coord_space: An optional coordinate space for the region being read.
                Defaults to ``None``, coordinate space will be inferred from transform.
            batch_size: The size of batched reads.
                Defaults to `unbatched`.
            partitions: If present, specifies that this is part of
                a partitioned read, and which part of the data to include.
            result_order: the order to return results, specified as a
                :class:`~options.ResultOrder` or its string value.
            value_filter: an optional value filter to apply to the results.
                The default of ``None`` represents no filter. Value filter
                syntax is implementation-defined; see the documentation
                for the particular SOMA implementation for details.

        Returns:
            A :class:`SpatialRead` with :class:`ReadIter` of :class:`pa.Table`s data.

        Lifecycle: experimental
        """
        raise NotImplementedError()

    def write(
        self,
        values: Union[pa.RecordBatch, pa.Table],
        *,
        platform_config: Optional[options.PlatformConfig] = None,
    ) -> Self:
        """Writes the data from an Arrow table to the persistent object.

        As duplicate index values are not allowed, index values already present
        in the object are overwritten and new index values are added.

        Args:
            values: An Arrow table containing all columns, including
                the index columns. The schema for the values must match
                the schema for the ``DataFrame``.

        Returns: ``self``, to enable method chaining.

        Lifecycle: experimental
        """
        _util.check_type("values", values, (pa.Table,))

        write_options: Union[TileDBCreateOptions, TileDBWriteOptions]
        sort_coords = None
        if isinstance(platform_config, TileDBCreateOptions):
            raise ValueError(
                "As of TileDB-SOMA 1.13, the write method takes "
                "TileDBWriteOptions instead of TileDBCreateOptions"
            )
        write_options = TileDBWriteOptions.from_platform_config(platform_config)
        sort_coords = write_options.sort_coords

        clib_dataframe = self._handle._handle

        for batch in values.to_batches():
            clib_dataframe.write(batch, sort_coords or False)

        if write_options.consolidate_and_vacuum:
            clib_dataframe.consolidate_and_vacuum()

        return self

    # Metadata operations

    @property
    def schema(self) -> pa.Schema:
        """The schema of the data in this dataframe.

        Lifecycle: experimental
        """
        return self._handle.schema

    @property
    def index_column_names(self) -> Tuple[str, ...]:
        """The names of the index (dimension) columns.

        Lifecycle: experimental
        """
        return tuple(self._tiledb_dim_names())

    @property
    def axis_names(self) -> Tuple[str, ...]:
        """The names of the axes of the coordinate space the data is defined on.

        Lifecycle: experimental
        """
        return tuple(self._axis_names)

    @property
    def domain(self) -> Tuple[Tuple[Any, Any], ...]:
        """The allowable range of values in each index column.

        Returns: a tuple of minimum and maximum values, inclusive,
            storable on each index column of the dataframe.

        Lifecycle: experimental
        """
        return tuple(self._domain())
