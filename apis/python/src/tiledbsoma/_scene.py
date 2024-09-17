# Copyright (c) 2024 TileDB, Inc.
#
# Licensed under the MIT License.

"""Implementation of a SOMA Scene."""

from typing import Optional, Sequence, Union

from somacore import coordinates, scene

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
    """SOMAScene is a specialized SOMACollection, for managing spatial assets.
    The SOMAScene defines a local coordinate system in physical space. The
    collection will provide operations for getting, setting, and transforming
    between coordinate systems. The SOMAScene contains the following reserved
    components.

    Attributes:
        img [SOMACollection[SOMAImageCollection]]:
            A collection of images (single and multi-resolution).
        obsl (SOMACollection[SOMAGeometryDataFrame | SOMADataFrame |  SOMAImageCollection]):
            A collection of spatial arrays and collections. The soma_joinid in
            these arrays corresponds to observations.
        varl (SOMACollection[SOMAGeometryNDArray | SOMAPointNDArray | SOMAImageCollection]):
            A collection of collections for spatial arrays on the SOMAMeasurements.
            The top level collection is indexed by measurement name. The soma_joinid
            in the contained arrays correspond to the features of the corresponding
            measurement.

    Lifecycle:
        Experimental.
    """

    __slots__ = ("_coordinate_space",)
    _wrapper_type = _tdb_handles.SceneWrapper

    _subclass_constrained_soma_types = {
        "img": ("SOMACollection",),
        "obsl": ("SOMACollection",),
        "varl": ("SOMACollection",),
    }

    @property
    def coordinate_space(self) -> Optional[coordinates.CoordinateSpace]:
        """Coordinate system for this scene.
        Lifecycle: experimental
        """
        return self._coordinate_space

    @coordinate_space.setter
    def coordinate_space(self, value: coordinates.CoordinateSpace) -> None:
        self._coordinate_space = value

    def register_geometry_dataframe(
        self,
        key: str,
        transform: coordinates.CoordinateTransform,
        *,
        subcollection: Union[str, Sequence[str]] = "obsl",
        coordinate_space: Optional[coordinates.CoordinateSpace] = None,
    ):
        # ) -> _GeometryDataFrame:
        """Adds the coordinate transform for the scene coordinate space to
        a point cloud stored in the scene.
        If the subcollection the geometry dataframe is inside of is more than one
        layer deep, the input should be provided as a sequence of names. For example,
        to register a geometry dataframe named  "transcripts" in the "var/RNA"
        collection::
            scene.register_geometry_dataframe(
                'transcripts', transform, subcollection=['var', 'RNA'],
            )
        Args:
            key: The name of the geometry dataframe.
            transform: The coordinate transformation from the scene to the dataframe.
            subcollection: The name, or sequence of names, of the subcollection the
                dataframe is stored in. Defaults to ``'obsl'``.
            coordinate_space: Optional coordinate space for the dataframe. This will
                replace the existing coordinate space of the dataframe.
        Returns:
            The registered geometry dataframe in write mode.
        Lifecycle: experimental
        """
        raise NotImplementedError()

    def register_multiscale_image(
        self,
        key: str,
        transform: coordinates.CoordinateTransform,
        *,
        subcollection: Union[str, Sequence[str]] = "img",
        coordinate_space: Optional[coordinates.CoordinateSpace] = None,
    ):
        # ) -> _MultiscaleImage:
        """Adds the coordinate transform for the scene coordinate space to
        a multiscale image stored in the scene.
        The transform to the multiscale image must be to the coordinate space
        defined on the reference level for the image. In most cases, this will be
        the level ``0`` image.
        Args:
            key: The name of the multiscale image.
            transform: The coordinate transformation from the scene to the reference
                level of the multiscale image.
            subcollection: The name, or sequence of names, of the subcollection the
                image is stored in. Defaults to ``'img'``.
            coordinate_space: Optional coordinate space for the image. This will
                replace the existing coordinate space of the multiscale image.
        Returns:
            The registered multiscale image in write mode.
        Lifecycle: experimental
        """
        raise NotImplementedError()

    def register_point_cloud(
        self,
        key: str,
        transform: coordinates.CoordinateTransform,
        *,
        subcollection: Union[str, Sequence[str]] = "obsl",
        coordinate_space: Optional[coordinates.CoordinateSpace] = None,
    ):
        # ) -> _PointCloud:
        """Adds the coordinate transform for the scene coordinate space to
        a point cloud stored in the scene.
        If the subcollection the point cloud is inside of is more than one
        layer deep, the input should be provided as a sequence of names. For example,
        to register a point named `transcripts` in the `var/RNA`
        collection::
            scene.register_point_cloud(
                'transcripts', transform, subcollection=['var', 'RNA'],
            )
        Args:
            key: The name of the point cloud.
            transform: The coordinate transformation from the scene to the point cloud.
            subcollection: The name, or sequence of names, of the subcollection the
                point cloud is stored in. Defaults to ``'obsl'``.
            coordinate_space: Optional coordinate space for the point cloud. This will
                replace the existing coordinate space of the point cloud. Defaults to
                ``None``.
        Returns:
            The registered point cloud in write mode.
        Lifecycle: experimental
        """
        raise NotImplementedError()

    def get_transformation_to_geometry_dataframe(
        self, key: str, *, subcollection: Union[str, Sequence[str]] = "obsl"
    ):
        """Returns the coordinate transformation from the scene to a requested
        geometery dataframe.
        Args:
            key: The name of the geometry dataframe.
            subcollection: The name, or sequence of names, of the subcollection the
                dataframe is stored in. Defaults to ``'obsl'``.
        Returns:
            Coordinate transform from the scene to the requested dataframe.
        Lifecycle: experimental
        """
        raise NotImplementedError()

    def get_transformation_to_multiscale_image(
        self,
        key: str,
        *,
        subcollection: str = "img",
        level: Optional[Union[str, int]] = None,
    ) -> coordinates.CoordinateTransform:
        """Returns the coordinate transformation from the scene to a requested
        multiscale image.
        Args:
            key: The name of the multiscale image.
            subcollection: The name, or sequence of names, of the subcollection the
                dataframe is stored in. Defaults to ``'img'``.
            level: The level of the image to get the transformation to.
                Defaults to ``None`` -- the transformation will be to the reference
                level.
        Returns:
            Coordinate transform from the scene to the requested multiscale image.
        Lifecycle: experimental
        """
        raise NotImplementedError()

    def get_transformation_to_point_cloud(
        self, key: str, *, subcollection: str = "obsl"
    ) -> coordinates.CoordinateTransform:
        """Returns the coordinate transformation from the scene to a requested
        geometery dataframe.
        Args:
            key: The name of the point cloud.
            subcollection: The name, or sequence of names, of the subcollection the
                point cloud is stored in. Defaults to ``'obsl'``.
        Returns:
            Coordinate transform from the scene to the requested point cloud.
        Lifecycle: experimental
        """
        raise NotImplementedError()
