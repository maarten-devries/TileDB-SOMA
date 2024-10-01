/**
 * @file   unit_soma_sparse_ndarray.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2022 TileDB, Inc.
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
 * This file manages unit tests for the SOMASparseNDArray class
 */

#include "common.h"

TEST_CASE("SOMASparseNDArray: basic", "[SOMASparseNDArray]") {
    // Core uses domain & current domain like (0, 999); SOMA uses shape like
    // 1000. We want to carefully and explicitly test here that there aren't any
    // off-by-one errors.
    int64_t dim_max = 999;
    int64_t shape = 1000;

    auto use_current_domain = GENERATE(false, true);
    // TODO this could be formatted with fmt::format which is part of internal
    // header spd/log/fmt/fmt.h and should not be used. In C++20, this can be
    // replaced with std::format.
    std::ostringstream section;
    section << "- use_current_domain=" << use_current_domain;
    SECTION(section.str()) {
        auto ctx = std::make_shared<SOMAContext>();
        std::string uri = "mem://unit-test-sparse-ndarray-basic";
        std::string dim_name = "soma_dim_0";
        std::string attr_name = "soma_data";
        tiledb_datatype_t dim_tiledb_datatype = TILEDB_INT64;
        tiledb_datatype_t attr_tiledb_datatype = TILEDB_INT32;
        std::string dim_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            dim_tiledb_datatype);
        std::string attr_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            attr_tiledb_datatype);

        REQUIRE(!SOMASparseNDArray::exists(uri, ctx));

        std::vector<helper::DimInfo> dim_infos(
            {{.name = dim_name,
              .tiledb_datatype = dim_tiledb_datatype,
              .dim_max = dim_max,
              .string_lo = "N/A",
              .string_hi = "N/A",
              .use_current_domain = use_current_domain}});

        auto index_columns = helper::create_column_index_info(dim_infos);

        SOMASparseNDArray::create(
            uri,
            attr_arrow_format,
            ArrowTable(
                std::move(index_columns.first),
                std::move(index_columns.second)),
            ctx,
            PlatformConfig(),
            TimestampRange(0, 2));

        REQUIRE(SOMASparseNDArray::exists(uri, ctx));
        REQUIRE(!SOMADataFrame::exists(uri, ctx));
        REQUIRE(!SOMADenseNDArray::exists(uri, ctx));

        auto soma_sparse = SOMASparseNDArray::open(uri, OpenMode::read, ctx);
        REQUIRE(soma_sparse->uri() == uri);
        REQUIRE(soma_sparse->ctx() == ctx);
        REQUIRE(soma_sparse->type() == "SOMASparseNDArray");
        REQUIRE(soma_sparse->is_sparse() == true);
        REQUIRE(soma_sparse->soma_data_type() == attr_arrow_format);
        auto schema = soma_sparse->tiledb_schema();
        REQUIRE(schema->has_attribute(attr_name));
        REQUIRE(schema->array_type() == TILEDB_SPARSE);
        REQUIRE(schema->domain().has_dimension(dim_name));
        REQUIRE(soma_sparse->ndim() == 1);
        REQUIRE(soma_sparse->nnz() == 0);

        auto expect = std::vector<int64_t>({shape});
        REQUIRE(soma_sparse->shape() == expect);
        if (!use_current_domain) {
            REQUIRE(soma_sparse->maxshape() == expect);
        }

        soma_sparse->close();

        std::vector<int64_t> d0(10);
        for (int j = 0; j < 10; j++)
            d0[j] = j;
        std::vector<int32_t> a0(10, 1);

        soma_sparse->open(OpenMode::write);
        soma_sparse->set_column_data(dim_name, d0.size(), d0.data());
        soma_sparse->set_column_data(attr_name, a0.size(), a0.data());
        soma_sparse->write();
        soma_sparse->close();

        soma_sparse->open(OpenMode::read);
        while (auto batch = soma_sparse->read_next()) {
            auto arrbuf = batch.value();
            auto d0span = arrbuf->at(dim_name)->data<int64_t>();
            auto a0span = arrbuf->at(attr_name)->data<int32_t>();
            REQUIRE(d0 == std::vector<int64_t>(d0span.begin(), d0span.end()));
            REQUIRE(a0 == std::vector<int32_t>(a0span.begin(), a0span.end()));
        }
        soma_sparse->close();

        std::vector<int64_t> d0b({dim_max, dim_max + 1});
        std::vector<int64_t> a0b({30, 40});

        // Try out-of-bounds write before resize.
        // * Without current domain support: this should throw since it's
        //   outside the (immutable) doqain.
        // * With current domain support: this should throw since it's outside
        // the (mutable) current domain.
        soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
        soma_sparse->set_column_data(dim_name, d0b.size(), d0b.data());
        soma_sparse->set_column_data(attr_name, a0b.size(), a0b.data());
        REQUIRE_THROWS(soma_sparse->write());
        soma_sparse->close();

        if (!use_current_domain) {
            auto new_shape = std::vector<int64_t>({shape});

            soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
            // Without current-domain support: this should throw since
            // one cannot resize what has not been sized.
            REQUIRE(!soma_sparse->has_current_domain());
            REQUIRE_THROWS(soma_sparse->resize(new_shape));
            // Now set the shape
            soma_sparse->upgrade_shape(new_shape);
            soma_sparse->close();

            soma_sparse->open(OpenMode::read);
            REQUIRE(soma_sparse->has_current_domain());
            soma_sparse->close();

            soma_sparse->open(OpenMode::write);
            REQUIRE(soma_sparse->has_current_domain());
            // Should not fail since we're setting it to what it already is.
            soma_sparse->resize(new_shape);
            soma_sparse->close();

            soma_sparse = SOMASparseNDArray::open(uri, OpenMode::read, ctx);
            REQUIRE(soma_sparse->shape() == new_shape);
            soma_sparse->close();

        } else {
            auto new_shape = std::vector<int64_t>({shape * 2});

            soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
            // Should throw since this already has a shape (core current
            // domain).
            REQUIRE_THROWS(soma_sparse->upgrade_shape(new_shape));
            soma_sparse->resize(new_shape);
            soma_sparse->close();

            // Try out-of-bounds write after resize.
            soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
            soma_sparse->set_column_data(dim_name, d0b.size(), d0b.data());
            soma_sparse->set_column_data(attr_name, a0b.size(), a0b.data());
            // Implicitly checking for no throw
            soma_sparse->write();
            soma_sparse->close();

            soma_sparse->open(OpenMode::read);
            REQUIRE(soma_sparse->shape() == new_shape);
            soma_sparse->close();
        }
    }
}

TEST_CASE("SOMASparseNDArray: platform_config", "[SOMASparseNDArray]") {
    int64_t dim_max = 999;
    auto use_current_domain = GENERATE(false, true);
    // TODO this could be formatted with fmt::format which is part of internal
    // header spd/log/fmt/fmt.h and should not be used. In C++20, this can be
    // replaced with std::format.
    std::ostringstream section;
    section << "- use_current_domain=" << use_current_domain;
    SECTION(section.str()) {
        auto ctx = std::make_shared<SOMAContext>();
        std::string uri = "mem://unit-test-dataframe-platform-config";
        std::string dim_name = "soma_dim_0";
        tiledb_datatype_t dim_tiledb_datatype = TILEDB_INT64;
        tiledb_datatype_t attr_tiledb_datatype = TILEDB_INT32;
        std::string dim_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            dim_tiledb_datatype);
        std::string attr_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            attr_tiledb_datatype);

        PlatformConfig platform_config;
        platform_config.sparse_nd_array_dim_zstd_level = 6;

        std::vector<helper::DimInfo> dim_infos(
            {{.name = dim_name,
              .tiledb_datatype = dim_tiledb_datatype,
              .dim_max = dim_max,
              .string_lo = "N/A",
              .string_hi = "N/A",
              .use_current_domain = use_current_domain}});

        auto index_columns = helper::create_column_index_info(dim_infos);

        SOMASparseNDArray::create(
            uri,
            attr_arrow_format,
            ArrowTable(
                std::move(index_columns.first),
                std::move(index_columns.second)),
            ctx,
            platform_config);

        auto soma_dataframe = SOMASparseNDArray::open(uri, OpenMode::read, ctx);
        auto dim_filter = soma_dataframe->tiledb_schema()
                              ->domain()
                              .dimension(dim_name)
                              .filter_list()
                              .filter(0);
        REQUIRE(dim_filter.filter_type() == TILEDB_FILTER_ZSTD);
        REQUIRE(dim_filter.get_option<int32_t>(TILEDB_COMPRESSION_LEVEL) == 6);

        soma_dataframe->close();
    }
}

TEST_CASE("SOMASparseNDArray: metadata", "[SOMASparseNDArray]") {
    int64_t dim_max = 999;
    auto use_current_domain = GENERATE(false, true);
    // TODO this could be formatted with fmt::format which is part of internal
    // header spd/log/fmt/fmt.h and should not be used. In C++20, this can be
    // replaced with std::format.
    std::ostringstream section;
    section << "- use_current_domain=" << use_current_domain;
    SECTION(section.str()) {
        auto ctx = std::make_shared<SOMAContext>();

        std::string uri = "mem://unit-test-sparse-ndarray";
        std::string dim_name = "soma_dim_0";
        tiledb_datatype_t dim_tiledb_datatype = TILEDB_INT64;
        tiledb_datatype_t attr_tiledb_datatype = TILEDB_INT32;
        std::string dim_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            dim_tiledb_datatype);
        std::string attr_arrow_format = ArrowAdapter::tdb_to_arrow_type(
            attr_tiledb_datatype);

        std::vector<helper::DimInfo> dim_infos(
            {{.name = dim_name,
              .tiledb_datatype = dim_tiledb_datatype,
              .dim_max = dim_max,
              .string_lo = "N/A",
              .string_hi = "N/A",
              .use_current_domain = use_current_domain}});

        auto index_columns = helper::create_column_index_info(dim_infos);

        SOMASparseNDArray::create(
            uri,
            attr_arrow_format,
            ArrowTable(
                std::move(index_columns.first),
                std::move(index_columns.second)),
            ctx,
            PlatformConfig(),
            TimestampRange(0, 2));

        auto soma_sparse = SOMASparseNDArray::open(
            uri,
            OpenMode::write,
            ctx,
            {},
            ResultOrder::automatic,
            std::pair<uint64_t, uint64_t>(1, 1));

        int32_t val = 100;
        soma_sparse->set_metadata("md", TILEDB_INT32, 1, &val);
        soma_sparse->close();

        // Read metadata
        soma_sparse->open(OpenMode::read, TimestampRange(0, 2));
        REQUIRE(soma_sparse->metadata_num() == 3);
        REQUIRE(soma_sparse->has_metadata("soma_object_type"));
        REQUIRE(soma_sparse->has_metadata("soma_encoding_version"));
        REQUIRE(soma_sparse->has_metadata("md"));
        auto mdval = soma_sparse->get_metadata("md");
        REQUIRE(std::get<MetadataInfo::dtype>(*mdval) == TILEDB_INT32);
        REQUIRE(std::get<MetadataInfo::num>(*mdval) == 1);
        REQUIRE(
            *((const int32_t*)std::get<MetadataInfo::value>(*mdval)) == 100);
        soma_sparse->close();

        // md should not be available at (2, 2)
        soma_sparse->open(OpenMode::read, TimestampRange(2, 2));
        REQUIRE(soma_sparse->metadata_num() == 2);
        REQUIRE(soma_sparse->has_metadata("soma_object_type"));
        REQUIRE(soma_sparse->has_metadata("soma_encoding_version"));
        REQUIRE(!soma_sparse->has_metadata("md"));
        soma_sparse->close();

        // Metadata should also be retrievable in write mode
        soma_sparse->open(OpenMode::write, TimestampRange(0, 2));
        REQUIRE(soma_sparse->metadata_num() == 3);
        REQUIRE(soma_sparse->has_metadata("soma_object_type"));
        REQUIRE(soma_sparse->has_metadata("soma_encoding_version"));
        REQUIRE(soma_sparse->has_metadata("md"));
        mdval = soma_sparse->get_metadata("md");
        REQUIRE(
            *((const int32_t*)std::get<MetadataInfo::value>(*mdval)) == 100);

        // Delete and have it reflected when reading metadata while in write
        // mode
        soma_sparse->delete_metadata("md");
        mdval = soma_sparse->get_metadata("md");
        REQUIRE(!mdval.has_value());
        soma_sparse->close();

        // Confirm delete in read mode
        soma_sparse->open(OpenMode::read, TimestampRange(0, 2));
        REQUIRE(!soma_sparse->has_metadata("md"));
        REQUIRE(soma_sparse->metadata_num() == 2);
    }
}
void breakme() {
}

TEST_CASE(
    "SOMASparseNDArray: can_tiledbsoma_upgrade_shape", "[SOMASparseNDArray]") {
    int64_t dim_max = 999;

    auto ctx = std::make_shared<SOMAContext>();
    std::string uri = "mem://unit-test-sparse-ndarray-upgrade-shape";

    std::string dim_name = "soma_dim_0";
    tiledb_datatype_t dim_tiledb_datatype = TILEDB_INT64;
    tiledb_datatype_t attr_tiledb_datatype = TILEDB_INT32;
    std::string dim_arrow_format = ArrowAdapter::tdb_to_arrow_type(
        dim_tiledb_datatype);
    std::string attr_arrow_format = ArrowAdapter::tdb_to_arrow_type(
        attr_tiledb_datatype);

    std::vector<helper::DimInfo> dim_infos(
        {{.name = dim_name,
          .tiledb_datatype = dim_tiledb_datatype,
          .dim_max = dim_max,
          .string_lo = "N/A",
          .string_hi = "N/A",
          .use_current_domain = false}});

    auto index_columns = helper::create_column_index_info(dim_infos);

    SOMASparseNDArray::create(
        uri,
        attr_arrow_format,
        ArrowTable(
            std::move(index_columns.first), std::move(index_columns.second)),
        ctx);

    auto soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
    REQUIRE(soma_sparse->has_current_domain() == false);

    // For old-style arrays, from before the current-domain feature:
    // * The shape specified at create becomes the core (max) domain
    //   o Recall that the core domain is immutable
    // * There is no current domain set
    //   o A current domain can be applied to it, up to <= (max) domain
    auto dom = soma_sparse->soma_domain_slot<int64_t>(dim_name);
    auto mxd = soma_sparse->soma_maxdomain_slot<int64_t>(dim_name);
    REQUIRE(dom == mxd);
    REQUIRE(dom.first == 0);
    REQUIRE(dom.second == dim_max);

    breakme();
    std::vector<int64_t> newshape_wrong_dims({dim_max, 12});
    std::vector<int64_t> newshape_too_big({dim_max + 10});
    std::vector<int64_t> newshape_good({40});

    auto check = soma_sparse->can_upgrade_shape(newshape_wrong_dims);
    REQUIRE(check.first == false);
    REQUIRE(
        check.second ==
        "cannot tiledbsoma_upgrade_shape: provided shape has ndim 2, while the "
        "array has 1");

    check = soma_sparse->can_upgrade_shape(newshape_too_big);
    REQUIRE(check.first == false);
    REQUIRE(
        check.second ==
        "cannot tiledbsoma_upgrade_shape for soma_dim_0: new 1009 < maxshape "
        "1000");

    check = soma_sparse->can_upgrade_shape(newshape_good);
    REQUIRE(check.first == true);
    REQUIRE(check.second == "");
}

TEST_CASE("SOMASparseNDArray: can_resize", "[SOMASparseNDArray]") {
    int64_t dim_max = 999;

    auto ctx = std::make_shared<SOMAContext>();
    std::string uri = "mem://unit-test-sparse-ndarray-resize";

    std::string dim_name = "soma_dim_0";
    tiledb_datatype_t dim_tiledb_datatype = TILEDB_INT64;
    tiledb_datatype_t attr_tiledb_datatype = TILEDB_INT32;
    std::string dim_arrow_format = ArrowAdapter::tdb_to_arrow_type(
        dim_tiledb_datatype);
    std::string attr_arrow_format = ArrowAdapter::tdb_to_arrow_type(
        attr_tiledb_datatype);

    std::vector<helper::DimInfo> dim_infos(
        {{.name = dim_name,
          .tiledb_datatype = dim_tiledb_datatype,
          .dim_max = dim_max,
          .string_lo = "N/A",
          .string_hi = "N/A",
          .use_current_domain = true}});

    auto index_columns = helper::create_column_index_info(dim_infos);

    SOMASparseNDArray::create(
        uri,
        attr_arrow_format,
        ArrowTable(
            std::move(index_columns.first), std::move(index_columns.second)),
        ctx);

    auto soma_sparse = SOMASparseNDArray::open(uri, OpenMode::write, ctx);
    REQUIRE(soma_sparse->has_current_domain() == true);

    // For new-style arrays, with the current-domain feature:
    // * The shape specified at create becomes the core current domain
    //   o Recall that the core current domain is mutable, up tp <= (max) domain
    // * The core (max) domain is huge
    //   o Recall that the core max domain is immutable
    auto dom = soma_sparse->soma_domain_slot<int64_t>(dim_name);
    auto mxd = soma_sparse->soma_maxdomain_slot<int64_t>(dim_name);
    REQUIRE(dom != mxd);
    REQUIRE(dom.first == 0);
    REQUIRE(dom.second == dim_max);

    std::vector<int64_t> newshape_wrong_dims({dim_max, 12});
    std::vector<int64_t> newshape_too_small({40});
    std::vector<int64_t> newshape_good({2000});

    auto check = soma_sparse->can_resize(newshape_wrong_dims);
    REQUIRE(check.first == false);
    REQUIRE(
        check.second ==
        "cannot resize: provided shape has ndim 2, while the array has 1");

    check = soma_sparse->can_resize(newshape_too_small);
    REQUIRE(check.first == false);
    REQUIRE(
        check.second ==
        "cannot resize for soma_dim_0: new 40 < existing shape 1000");

    check = soma_sparse->can_resize(newshape_good);
    REQUIRE(check.first == true);
    REQUIRE(check.second == "");
}
