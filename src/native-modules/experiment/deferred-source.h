#pragma once

#include <string>
#include <vector>
#include "base-types.h"
#include "source-core.h"

namespace cm::deferred_source {

enum class MaterializeMode {
  Full,
  NcListOnly,
};

auto from_primary(const SourceData& src, const std::string& name, index_t idx)
    -> DeferredSourceData;

auto combine(const DeferredSourceData& first, const DeferredSourceData& second)
    -> DeferredSourceData;

auto materialize(const DeferredSourceData& combo,
    MaterializeMode mode = MaterializeMode::Full) -> SourceData;

auto materialize_selected(const std::vector<index_t>& combo_indices,
    const std::vector<DeferredSourceDataList>& combo_lists,
    MaterializeMode mode = MaterializeMode::Full) -> SourceData;

}  // namespace cm::deferred_source
