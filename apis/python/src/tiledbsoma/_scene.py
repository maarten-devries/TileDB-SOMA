# Copyright (c) 2024 TileDB, Inc.
#
# Licensed under the MIT License.

"""Implementation of a SOMA Scene."""

from typing import Union

from somacore import scene

from . import _tdb_handles
from ._collection import Collection, CollectionBase
from ._dataframe import DataFrame
from ._dense_nd_array import DenseNDArray
from ._soma_object import AnySOMAObject
from ._sparse_nd_array import SparseNDArray


class Scene(  # type: ignore[misc]  # __eq__ false positive
    CollectionBase[AnySOMAObject],
    scene.Scene[  # type: ignore[type-var]
        DataFrame,
        Collection[
            Union[DenseNDArray, SparseNDArray]
        ],  # not just NDArray since NDArray does not have a common `read`
        AnySOMAObject,
    ],
):
    """A collection subtype of spatial assets where all assets in one Scene 
    correspond to the same physical coordinate system.

    TODO description

    Lifecycle:
        Experimental.
    """

    __slots__ = ()
    _wrapper_type = _tdb_handles.SceneWrapper

    _subclass_constrained_soma_types = {
        "img": ("SOMACollection",),
        "obsl": ("SOMACollection",),
        "varl": ("SOMACollection",),
    }