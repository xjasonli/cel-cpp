// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_CEL_CPP_EVAL_COMPILER_RESOLVER_H_
#define THIRD_PARTY_CEL_CPP_EVAL_COMPILER_RESOLVER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "common/kind.h"
#include "common/type.h"
#include "common/type_reflector.h"
#include "common/value.h"
#include "runtime/function_overload_reference.h"
#include "runtime/function_registry.h"
#include "runtime/type_registry.h"

namespace google::api::expr::runtime {

// Resolver assists with finding functions and types from the associated
// registries within a container.
//
// container is used to construct the namespace lookup candidates.
// e.g. for "cel.dev" -> {"cel.dev.", "cel.", ""}
class Resolver {
 public:
  Resolver(absl::string_view container,
           const cel::FunctionRegistry& function_registry,
           const cel::TypeRegistry& type_registry,
           const cel::TypeReflector& type_reflector,
           bool resolve_qualified_type_identifiers = true);

  Resolver(const Resolver&) = delete;
  Resolver& operator=(const Resolver&) = delete;
  Resolver(Resolver&&) = delete;
  Resolver& operator=(Resolver&&) = delete;

  ~Resolver() = default;

  // FindConstant will return an enum constant value or a type value if one
  // exists for the given name. An empty handle will be returned if none exists.
  //
  // Since enums and type identifiers are specified as (potentially) qualified
  // names within an expression, there is the chance that the name provided
  // is a variable name which happens to collide with an existing enum or proto
  // based type name. For this reason, within parsed only expressions, the
  // constant should be treated as a value that can be shadowed by a runtime
  // provided value.
  absl::optional<cel::Value> FindConstant(absl::string_view name,
                                          int64_t expr_id) const;

  absl::StatusOr<absl::optional<std::pair<std::string, cel::Type>>> FindType(
      absl::string_view name, int64_t expr_id) const;

  // FindLazyOverloads returns the set, possibly empty, of lazy overloads
  // matching the given function signature.
  std::vector<cel::FunctionRegistry::LazyOverload> FindLazyOverloads(
      absl::string_view name, bool receiver_style,
      const std::vector<cel::Kind>& types, int64_t expr_id = -1) const;

  std::vector<cel::FunctionRegistry::LazyOverload> FindLazyOverloads(
      absl::string_view name, bool receiver_style, size_t arity,
      int64_t expr_id = -1) const;

  // FindOverloads returns the set, possibly empty, of eager function overloads
  // matching the given function signature.
  std::vector<cel::FunctionOverloadReference> FindOverloads(
      absl::string_view name, bool receiver_style,
      const std::vector<cel::Kind>& types, int64_t expr_id = -1) const;

  std::vector<cel::FunctionOverloadReference> FindOverloads(
      absl::string_view name, bool receiver_style, size_t arity,
      int64_t expr_id = -1) const;

  // FullyQualifiedNames returns the set of fully qualified names which may be
  // derived from the base_name within the specified expression container.
  std::vector<std::string> FullyQualifiedNames(absl::string_view base_name,
                                               int64_t expr_id = -1) const;

 private:
  absl::Span<const std::string> GetPrefixesFor(absl::string_view& name) const;

  std::vector<std::string> namespace_prefixes_;
  std::shared_ptr<const absl::flat_hash_map<std::string, cel::Value>>
      enum_value_map_;
  const cel::FunctionRegistry& function_registry_;
  const cel::TypeReflector& type_reflector_;

  bool resolve_qualified_type_identifiers_;
};

// ArgumentMatcher generates a function signature matcher for CelFunctions.
// TODO(issues/91): this is the same behavior as parsed exprs in the CPP
// evaluator (just check the right call style and number of arguments), but we
// should have enough type information in a checked expr to find a more
// specific candidate list.
inline std::vector<cel::Kind> ArgumentsMatcher(int argument_count) {
  std::vector<cel::Kind> argument_matcher(argument_count);
  for (int i = 0; i < argument_count; i++) {
    argument_matcher[i] = cel::Kind::kAny;
  }
  return argument_matcher;
}

}  // namespace google::api::expr::runtime

#endif  // THIRD_PARTY_CEL_CPP_EVAL_COMPILER_RESOLVER_H_
