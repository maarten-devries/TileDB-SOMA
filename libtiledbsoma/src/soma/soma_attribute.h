/**
 * @file   soma_attribute.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024-2025 TileDB, Inc.
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
 *   This file defines the SOMAAttribute class. SOMAAttribute extends SOMAColumn
 * and wraps a TileDB Attribute and optionally an enumeration associated with
 * the attribute. The purpose of this class is to provide a common interface
 * identical to TileDB dimensions and composite columns.
 */

#ifndef SOMA_ATTRIBUTE_H
#define SOMA_ATTRIBUTE_H

#include <algorithm>
#include <vector>

#include <tiledb/tiledb>
#include "soma_column.h"

namespace tiledbsoma {
using namespace tiledb;

class SOMAAttribute : public SOMAColumn {
   public:
    /**
     * Create a ``SOMAAttribute`` shared pointer from an Arrow schema
     */
    static std::shared_ptr<SOMAAttribute> create(
        std::shared_ptr<Context> ctx,
        ArrowSchema* schema,
        std::string_view type_metadata,
        PlatformConfig platform_config);

    SOMAAttribute(
        Attribute attribute,
        std::optional<Enumeration> enumeration = std::nullopt)
        : attribute(attribute)
        , enumeration(enumeration) {
    }

    inline std::string name() const override {
        return attribute.name();
    }

    inline bool isIndexColumn() const override {
        return false;
    }

    inline void select_columns(
        const std::unique_ptr<ManagedQuery>& query,
        bool if_not_empty = false) const override {
        query->select_columns(std::vector({attribute.name()}), if_not_empty);
    };

    inline soma_column_datatype_t type() const override {
        return soma_column_datatype_t::SOMA_COLUMN_ATTRIBUTE;
    }

    inline std::optional<tiledb_datatype_t> domain_type() const override {
        return std::nullopt;
    }

    inline std::optional<tiledb_datatype_t> data_type() const override {
        return attribute.type();
    }

    inline std::optional<std::vector<Dimension>> tiledb_dimensions() override {
        return std::nullopt;
    }

    inline std::optional<std::vector<Attribute>> tiledb_attributes() override {
        return std::vector({attribute});
    }

    inline std::optional<std::vector<Enumeration>> tiledb_enumerations()
        override {
        if (!enumeration.has_value()) {
            return std::nullopt;
        }

        return std::vector({enumeration.value()});
    }

    ArrowArray* arrow_domain_slot(
        const SOMAContext& ctx,
        Array& array,
        enum Domainish kind) const override;

    ArrowSchema* arrow_schema_slot(
        const SOMAContext& ctx, Array& array) override;

   private:
    void _set_dim_points(
        const std::unique_ptr<ManagedQuery>& query,
        const SOMAContext& ctx,
        const std::any& points) const override;

    void _set_dim_ranges(
        const std::unique_ptr<ManagedQuery>& query,
        const SOMAContext& ctx,
        const std::any& ranges) const override;

    void _set_current_domain_slot(
        NDRectangle& rectangle,
        std::span<const std::any> domain) const override;

    std::pair<bool, std::string> _can_set_current_domain_slot(
        std::optional<NDRectangle>& rectangle,
        std::span<const std::any> new_domain) const override;

    std::any _core_domain_slot() const override;

    std::any _non_empty_domain_slot(Array& array) const override;

    std::any _non_empty_domain_slot_opt(
        const SOMAContext& ctx, Array& array) const override;

    std::any _core_current_domain_slot(
        const SOMAContext& ctx, Array& array) const override;

    std::any _core_current_domain_slot(NDRectangle& ndrect) const override;

    Attribute attribute;
    std::optional<Enumeration> enumeration;
};
}  // namespace tiledbsoma

#endif