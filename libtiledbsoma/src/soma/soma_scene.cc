/**
 * @file   soma_scene.cc
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
 *   This file defines the SOMAScene class.
 */

#include "soma_scene.h"
#include "soma_collection.h"
#include "utils/common.h"

namespace tiledbsoma {
using namespace tiledb;

//===================================================================
//= public static
//===================================================================

void SOMAScene::create(
    std::string_view uri,
    std::shared_ptr<SOMAContext> ctx,
    const std::optional<SOMACoordinateSpace>& coordinate_space,
    std::optional<TimestampRange> timestamp) {
    try {
        std::filesystem::path scene_uri(uri);
        auto group = SOMAGroup::create(
            ctx, scene_uri.string(), "SOMAScene", timestamp);

        group->set_metadata(
            SPATIAL_ENCODING_VERSION_KEY,
            TILEDB_STRING_UTF8,
            static_cast<uint32_t>(SPATIAL_ENCODING_VERSION_VAL.size()),
            SPATIAL_ENCODING_VERSION_VAL.c_str(),
            true);

        if (coordinate_space.has_value()) {
            const auto coord_space_metadata = coordinate_space->to_string();
            group->set_metadata(
                SOMA_COORDINATE_SPACE_KEY,
                TILEDB_STRING_UTF8,
                static_cast<uint32_t>(coord_space_metadata.size()),
                coord_space_metadata.c_str(),
                true);
        }

        group->close();
    } catch (TileDBError& e) {
        throw TileDBSOMAError(e.what());
    }
}

std::unique_ptr<SOMAScene> SOMAScene::open(
    std::string_view uri,
    OpenMode mode,
    std::shared_ptr<SOMAContext> ctx,
    std::optional<TimestampRange> timestamp) {
    try {
        auto group = std::make_unique<SOMAScene>(mode, uri, ctx, timestamp);

        if (!group->check_type("SOMAScene")) {
            throw TileDBSOMAError(
                "[SOMAScene::open] Object is not a SOMAScene");
        }

        auto coord_space_metadata = group->get_metadata(
            SOMA_COORDINATE_SPACE_KEY);
        if (coord_space_metadata.has_value()) {
            group->coord_space_ = SOMACoordinateSpace::from_metadata(
                std::get<0>(coord_space_metadata.value()),
                std::get<1>(coord_space_metadata.value()),
                std::get<2>(coord_space_metadata.value()));
        }

        return group;
    } catch (TileDBError& e) {
        throw TileDBSOMAError(e.what());
    }
}

std::shared_ptr<SOMACollection> SOMAScene::img() {
    if (img_ == nullptr) {
        img_ = SOMACollection::open(
            (std::filesystem::path(uri()) / "img").string(),
            OpenMode::read,
            ctx(),
            timestamp());
    }
    return img_;
}

std::shared_ptr<SOMACollection> SOMAScene::obsl() {
    if (obsl_ == nullptr) {
        obsl_ = SOMACollection::open(
            (std::filesystem::path(uri()) / "obsl").string(),
            OpenMode::read,
            ctx(),
            timestamp());
    }
    return obsl_;
}

std::shared_ptr<SOMACollection> SOMAScene::varl() {
    if (varl_ == nullptr) {
        varl_ = SOMACollection::open(
            (std::filesystem::path(uri()) / "varl").string(),
            OpenMode::read,
            ctx(),
            timestamp());
    }
    return varl_;
}

}  // namespace tiledbsoma
