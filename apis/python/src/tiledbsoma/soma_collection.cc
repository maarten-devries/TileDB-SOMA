/**
 * @file   soma_collection.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * This file defines the SOMACollection bindings.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <tiledbsoma/tiledbsoma>

#include "common.h"

namespace libtiledbsomacpp {

namespace py = pybind11;
using namespace py::literals;
using namespace tiledbsoma;

void load_soma_collection(py::module& m) {
    py::class_<SOMACollection, SOMAGroup, SOMAObject>(m, "SOMACollection")
        .def_static(
            "open",
            py::overload_cast<
                std::string_view,
                OpenMode,
                std::shared_ptr<SOMAContext>,
                std::optional<std::pair<uint64_t, uint64_t>>>(
                &SOMACollection::open),
            "uri"_a,
            py::kw_only(),
            "mode"_a,
            "context"_a,
            "timestamp"_a = py::none(),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__iter__",
            [](SOMACollection& collection) {
                return py::make_iterator(collection.begin(), collection.end());
            },
            py::keep_alive<0, 1>())
        .def("get", &SOMACollection::get);

    py::class_<SOMAExperiment, SOMACollection, SOMAGroup, SOMAObject>(
        m, "SOMAExperiment")
        .def_static(
            "open",
            &SOMAExperiment::open,
            "uri"_a,
            py::kw_only(),
            "mode"_a,
            "context"_a,
            "timestamp"_a = py::none(),
            py::call_guard<py::gil_scoped_release>());

    py::class_<SOMAMeasurement, SOMACollection, SOMAGroup, SOMAObject>(
        m, "SOMAMeasurement")
        .def_static(
            "open",
            &SOMAMeasurement::open,
            "uri"_a,
            py::kw_only(),
            "mode"_a,
            "context"_a,
            "timestamp"_a = py::none(),
            py::call_guard<py::gil_scoped_release>());

    py::class_<SOMAScene, SOMACollection, SOMAGroup, SOMAObject>(m, "SOMAScene")
        .def_static(
            "create",
            [](std::shared_ptr<SOMAContext> ctx,
               std::string_view uri,
               std::optional<std::vector<std::string>> axis_names,
               std::optional<std::vector<std::optional<std::string>>>
                   axis_units,
               std::optional<TimestampRange> timestamp) {
                if (axis_units.has_value() && !axis_names.has_value()) {
                    throw TileDBSOMAError(
                        "Cannot provide axis units without axis names.");
                }
                std::optional<SOMACoordinateSpace> coord_space{std::nullopt};
                if (axis_names.has_value()) {
                    if (axis_units.has_value()) {
                        coord_space = SOMACoordinateSpace(
                            axis_names.value(), axis_units.value());
                    } else {
                        coord_space = SOMACoordinateSpace(axis_names.value());
                    }
                }
                try {
                    SOMAScene::create(uri, ctx, coord_space, timestamp);
                } catch (const std::exception& e) {
                    TPY_ERROR_LOC(e.what());
                }
            },
            py::kw_only(),
            "ctx"_a,
            "uri"_a,
            "axis_names"_a,
            "axis_units"_a,
            "timestamp"_a = py::none())
        .def_static(
            "open",
            &SOMAScene::open,
            "uri"_a,
            py::kw_only(),
            "mode"_a,
            "context"_a,
            "timestamp"_a = py::none(),
            py::call_guard<py::gil_scoped_release>());

    py::class_<SOMAMultiscaleImage, SOMACollection, SOMAGroup, SOMAObject>(
        m, "SOMAMultiscaleImage")
        .def_static(
            "open",
            &SOMAMultiscaleImage::open,
            "uri"_a,
            py::kw_only(),
            "mode"_a,
            "context"_a,
            "timestamp"_a = py::none(),
            py::call_guard<py::gil_scoped_release>());
}
}  // namespace libtiledbsomacpp
