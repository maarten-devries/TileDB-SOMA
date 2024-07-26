#
# FindTileDB_EP.cmake
#
#
# The MIT License
#
# Copyright (c) 2018 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Finds the TileDB library, installing with an ExternalProject as necessary.

# If TileDB was installed as an EP, need to search the EP install path also.
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${EP_INSTALL_PREFIX}")

if (FORCE_BUILD_TILEDB)
  find_package(TileDB CONFIG PATHS ${EP_INSTALL_PREFIX} NO_DEFAULT_PATH)
else()
  find_package(TileDB CONFIG)
endif()

if (TILEDB_FOUND)
  get_target_property(TILEDB_LIB TileDB::tiledb_shared IMPORTED_LOCATION_RELEASE)
  # If release build location not found, check for debug build location
  if (TILEDB_LIB MATCHES "NOTFOUND")
    get_target_property(TILEDB_LIB TileDB::tiledb_shared IMPORTED_LOCATION_DEBUG)
  endif()
  message(STATUS "Found TileDB: ${TILEDB_LIB}")
else()
  if (SUPERBUILD)
    message(STATUS "Adding TileDB as an external project")
    if (TILEDB_S3 STREQUAL "OFF")
      message(STATUS "TileDB will be built WITHOUT S3 support")
    endif()

    # Try to download prebuilt artifacts unless the user specifies to build from source.
    # The TileDB Embedded version specified here will be linked to the libtiledbsoma native lib
    # loaded by the Python and R tiledbsoma packages. Those packages -also- use TileDB-Py and
    # TileDB-R, each of which links their own 'copy' of TileDB Embedded, whose version we don't
    # control here. Ideally the TileDB Embedded versions should match! The show_package_versions()
    # helper function in each package can help to diagnose any mismatch.
    # NB When updating the pinned URLs here, please also update in file apis/r/tools/get_tarball.R
    if(DOWNLOAD_TILEDB_PREBUILT)
        if (WIN32) # Windows
          SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-windows-x86_64-2.25.0-bbcbd3f.zip")
          SET(DOWNLOAD_SHA1 "b2b8cdfefbcbb07296c86e52cb9a2979ef455d2a")
        elseif(APPLE) # OSX

          # Status quo as of 2023-05-18:
          # * GitHub Actions does not have MacOS arm64 hardware available -- tracked at
          #   https://github.com/github/roadmap/issues/528
          # * The best we can do is cross-compile for arm64 while actually running on x86_64
          # * When we're invoked from python setup.py:
          #   o CMAKE_OSX_ARCHITECTURES is x86_64 or arm64
          #   o CMAKE_SYSTEM_PROCESSOR is x86_64 in either case
          #   o We must download the tiledb artifacts for the cross-compile architecture (x86_64 or arm64)
          #   o Therefore we must respect CMAKE_OSX_ARCHITECTURES not CMAKE_SYSTEM_PROCESSOR
          # * When we're invoked from R configure and src/Makevars.in:
          #   o CMAKE_OSX_ARCHITECTURES is unset
          #   o CMAKE_SYSTEM_PROCESSOR is x86_64

          if (CMAKE_OSX_ARCHITECTURES STREQUAL x86_64)
            SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-macos-x86_64-2.25.0-bbcbd3f.tar.gz")
            SET(DOWNLOAD_SHA1 "9c799d7de64e78d9e95cfa7db95b569cd3d3c8b0")
          elseif (CMAKE_OSX_ARCHITECTURES STREQUAL arm64)
            SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-macos-arm64-2.25.0-bbcbd3f.tar.gz")
            SET(DOWNLOAD_SHA1 "c6a1ab5213c36e53dd8e6da87c18b8bc695c532d")
          elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
            SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-macos-x86_64-2.25.0-bbcbd3f.tar.gz")
            SET(DOWNLOAD_SHA1 "9c799d7de64e78d9e95cfa7db95b569cd3d3c8b0")
          elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
            SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-macos-arm64-2.25.0-bbcbd3f.tar.gz")
            SET(DOWNLOAD_SHA1 "c6a1ab5213c36e53dd8e6da87c18b8bc695c532d")
          endif()

        else() # Linux
          SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.25.0/tiledb-linux-x86_64-2.25.0-bbcbd3f.tar.gz")
          SET(DOWNLOAD_SHA1 "1d789af5d88ce09edf60d16ea3988631e780252a")
        endif()

        ExternalProject_Add(ep_tiledb
                PREFIX "externals"
                URL ${DOWNLOAD_URL}
                URL_HASH SHA1=${DOWNLOAD_SHA1}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                UPDATE_COMMAND ""
                PATCH_COMMAND ""
                TEST_COMMAND ""
                INSTALL_COMMAND
                    ${CMAKE_COMMAND} -E copy_directory ${EP_BASE}/src/ep_tiledb ${EP_INSTALL_PREFIX}
                LOG_DOWNLOAD TRUE
                LOG_CONFIGURE FALSE
                LOG_BUILD FALSE
                LOG_INSTALL FALSE
                )
    else() # Build from source
        ExternalProject_Add(ep_tiledb
          PREFIX "externals"
          URL "https://github.com/TileDB-Inc/TileDB/archive/2.25.0.zip"
          URL_HASH SHA1=35418d02cefe2721e3e3b6a3d52505359f4b3e21
          DOWNLOAD_NAME "tiledb.zip"
          CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EP_INSTALL_PREFIX}
            -DCMAKE_PREFIX_PATH=${EP_INSTALL_PREFIX}
            -DTILEDB_S3=${TILEDB_S3}
            -DTILEDB_AZURE=${TILEDB_AZURE}
            -DTILEDB_GCS=${TILEDB_GCS}
            -DTILEDB_HDFS=${TILEDB_HDFS}
            -DTILEDB_SERIALIZATION=${TILEDB_SERIALIZATION}
            -DTILEDB_WERROR=${TILEDB_WERROR}
            -DTILEDB_REMOVE_DEPRECATIONS=${TILEDB_REMOVE_DEPRECATIONS}
            -DTILEDB_VERBOSE=${TILEDB_VERBOSE}
            -DTILEDB_TESTS=OFF
            -DCMAKE_BUILD_TYPE=$<CONFIG>
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            $<$<BOOL:${CMAKE_TOOLCHAIN_FILE}>:-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          UPDATE_COMMAND ""
          INSTALL_COMMAND
            ${CMAKE_COMMAND} --build . --target install-tiledb
          LOG_DOWNLOAD TRUE
          LOG_CONFIGURE TRUE
          LOG_BUILD TRUE
          LOG_INSTALL TRUE
        )
    endif()

    list(APPEND FORWARD_EP_CMAKE_ARGS -DEP_TILEDB_BUILT=TRUE)
    list(APPEND EXTERNAL_PROJECTS ep_tiledb)
  else()
    message(FATAL_ERROR "Unable to find TileDB library.")
  endif()
endif()

if (EP_TILEDB_BUILT AND TARGET TileDB::tiledb_shared)
  include(TileDBCommon)
  install_target_libs(TileDB::tiledb_shared)
endif()
