#ifndef SOMA_DIMENSION_H
#define SOMA_DIMENSION_H

#include <algorithm>
#include <vector>

#include <tiledb/tiledb>
#include "soma_column.h"

namespace tiledbsoma {

using namespace tiledb;

class SOMADimension : public virtual SOMAColumn {
   public:
    static std::shared_ptr<SOMADimension> create(
        std::shared_ptr<Context> ctx,
        ArrowSchema* schema,
        ArrowArray* array,
        const std::string& soma_type,
        std::string_view type_metadata,
        PlatformConfig platform_config,
        bool& has_current_domain);

    SOMADimension(Dimension dimension)
        : dimension(dimension) {
    }

    inline std::string name() const {
        return dimension.name();
    }

    inline bool isIndexColumn() const {
        return true;
    }

    inline virtual void select_columns(
        const std::unique_ptr<ManagedQuery>& query,
        bool if_not_empty = false) const override {
        query->select_columns(std::vector({dimension.name()}), if_not_empty);
    };

    inline soma_column_datatype_t type() const {
        return soma_column_datatype_t::SOMA_COLUMN_DIMENSION;
    }

    inline std::optional<tiledb_datatype_t> domain_type() const {
        return dimension.type();
    }

    inline std::optional<tiledb_datatype_t> data_type() const {
        return std::nullopt;
    }

    inline std::optional<std::vector<Dimension>> tiledb_dimensions() {
        return std::vector({dimension});
    }

    inline std::optional<std::vector<Attribute>> tiledb_attributes() {
        return std::nullopt;
    }

    inline virtual std::optional<std::vector<Enumeration>>
    tiledb_enumerations() {
        return std::nullopt;
    }

    virtual ArrowArray* arrow_domain_slot(
        const SOMAContext& ctx,
        Array& array,
        enum Domainish kind) const override;

    virtual ArrowSchema* arrow_schema_slot(
        const SOMAContext& ctx, Array& array) override;

   protected:
    virtual void _set_dim_points(
        const std::unique_ptr<ManagedQuery>& query,
        const SOMAContext& ctx,
        const std::any& ranges) const override;

    virtual void _set_dim_ranges(
        const std::unique_ptr<ManagedQuery>& query,
        const SOMAContext& ctx,
        const std::any& ranges) const override;

    virtual void _set_current_domain_slot(
        NDRectangle& rectangle,
        const std::vector<const void*>& domain) const override;

    virtual std::any _core_domain_slot() const override;

    virtual std::any _non_empty_domain_slot(Array& array) const override;

    virtual std::any _core_current_domain_slot(
        const SOMAContext& ctx, Array& array) const override;

   private:
    Dimension dimension;
};
}  // namespace tiledbsoma

#endif