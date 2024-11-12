#ifndef SOMA_ATTRIBUTE
#define SOMA_ATTRIBUTE

#include <algorithm>
#include <vector>

#include <tiledb/tiledb>
#include "soma_column.h"

namespace tiledbsoma {
using namespace tiledb;

class SOMAAttribute : public virtual SOMAColumn {
   public:
    static std::shared_ptr<SOMAAttribute> create(
        std::shared_ptr<Context> ctx,
        ArrowSchema* schema,
        std::string_view type_metadata,
        PlatformConfig platform_config);

    SOMAAttribute(const Context& ctx, Attribute attribute)
        : SOMAColumn(ctx)
        , attribute(attribute) {
    }

    inline std::string domain_to_str() const {
        return "";
    }

    virtual inline std::string name() const {
        return attribute.name();
    }

    virtual inline bool isIndex() const {
        return false;
    }

    inline std::optional<std::vector<Dimension>> tiledb_dimensions() {
        return std::nullopt;
    }

    inline std::optional<std::vector<Attribute>> tiledb_attributes() {
        return std::vector({attribute});
    }

    inline virtual std::optional<std::vector<Enumeration>>
    tiledb_enumerations() {
        if (!enumeration.has_value()) {
            return std::nullopt;
        }

        return std::vector({enumeration.value()});
    }

    inline virtual bool has_current_domain() {
        return false;
    }

   private:
    virtual void _set_dim_ranges(
        const std::unique_ptr<ManagedQuery>& query,
        const std::any& ranges) const;

    virtual void _set_current_domain_slot(
        NDRectangle& rectangle, const std::vector<const void*>& domain) const;

    virtual std::any _core_domain_slot() const;

    virtual std::any _non_empty_domain_slot(Array& array) const;

    virtual std::any _core_current_domain_slot(Array& array) const;

    Attribute attribute;
    std::optional<Enumeration> enumeration;
};
}  // namespace tiledbsoma

#endif