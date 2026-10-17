// Copyright 2023 Google LLC
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

#include "common/function_descriptor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

#include "absl/base/macros.h"
#include "absl/functional/overload.h"
#include "absl/types/span.h"
#include "common/ast.h"
#include "common/kind.h"

namespace cel {

bool FunctionDescriptor::ShapeMatches(bool receiver_style,
                                      absl::Span<const Kind> types) const {
  if (this->receiver_style() != receiver_style) {
    return false;
  }

  if (this->types().size() != types.size()) {
    return false;
  }

  for (size_t i = 0; i < this->types().size(); i++) {
    Kind this_type = this->types()[i];
    Kind other_type = types[i];
    if (this_type != Kind::kAny && other_type != Kind::kAny &&
        this_type != other_type) {
      return false;
    }
  }
  return true;
}

bool FunctionDescriptor::operator==(const FunctionDescriptor& other) const {
  return name() == other.name() && receiver_style() == other.receiver_style() &&
         types().size() == other.types().size() &&
         std::equal(types().begin(), types().end(), other.types().begin());
}

bool FunctionDescriptor::operator<(const FunctionDescriptor& other) const {
  if (name() < other.name()) {
    return true;
  }
  if (name() != other.name()) {
    return false;
  }
  if (receiver_style() < other.receiver_style()) {
    return true;
  }
  if (receiver_style() != other.receiver_style()) {
    return false;
  }
  auto lhs_begin = types().begin();
  auto lhs_end = types().end();
  auto rhs_begin = other.types().begin();
  auto rhs_end = other.types().end();
  while (lhs_begin != lhs_end && rhs_begin != rhs_end) {
    if (*lhs_begin < *rhs_begin) {
      return true;
    }
    if (!(*lhs_begin == *rhs_begin)) {
      return false;
    }
    lhs_begin++;
    rhs_begin++;
  }
  if (lhs_begin == lhs_end && rhs_begin == rhs_end) {
    // Neither has any elements left, they are equal.
    return false;
  }
  if (lhs_begin == lhs_end) {
    // Left has no more elements. Right is greater.
    return true;
  }
  // Right has no more elements. Left is greater.
  ABSL_ASSERT(rhs_begin == rhs_end);
  return false;
}

namespace {

[[nodiscard]]
Kind TypeSpecToKind(const TypeSpec& spec) {
  return std::visit(
      absl::Overload(
          [](UnsetTypeSpec) -> Kind { return Kind::kError; },
          [](const DynTypeSpec& spec) -> Kind { return Kind::kDyn; },
          [](const NullTypeSpec& spec) -> Kind { return Kind::kNull; },
          [](const PrimitiveType& spec) -> Kind {
            switch (spec) {
              case PrimitiveType::kPrimitiveTypeUnspecified:
                return Kind::kError;
              case PrimitiveType::kBool:
                return Kind::kBool;
              case PrimitiveType::kInt64:
                return Kind::kInt;
              case PrimitiveType::kUint64:
                return Kind::kUint;
              case PrimitiveType::kDouble:
                return Kind::kDouble;
              case PrimitiveType::kString:
                return Kind::kString;
              case PrimitiveType::kBytes:
                return Kind::kBytes;
            }
          },
          [](const PrimitiveTypeWrapper& spec) -> Kind {
            switch (spec.type()) {
              case PrimitiveType::kPrimitiveTypeUnspecified:
                return Kind::kError;
              case PrimitiveType::kBool:
                return Kind::kBoolWrapper;
              case PrimitiveType::kInt64:
                return Kind::kIntWrapper;
              case PrimitiveType::kUint64:
                return Kind::kUintWrapper;
              case PrimitiveType::kDouble:
                return Kind::kDoubleWrapper;
              case PrimitiveType::kString:
                return Kind::kStringWrapper;
              case PrimitiveType::kBytes:
                return Kind::kBytesWrapper;
            }
          },
          [](const WellKnownTypeSpec& spec) -> Kind {
            switch (spec) {
              case cel::WellKnownTypeSpec::kWellKnownTypeUnspecified:
                return Kind::kError;
              case cel::WellKnownTypeSpec::kDuration:
                return Kind::kError;
              case cel::WellKnownTypeSpec::kTimestamp:
                return Kind::kTimestamp;
              case cel::WellKnownTypeSpec::kAny:
                return Kind::kAny;
            }
          },
          [](const ListTypeSpec& spec) -> Kind { return Kind::kList; },
          [](const MapTypeSpec& spec) -> Kind { return Kind::kMap; },
          [](const FunctionTypeSpec& spec) -> Kind { return Kind::kFunction; },
          [](const MessageTypeSpec& spec) -> Kind { return Kind::kStruct; },
          [](const ParamTypeSpec& spec) -> Kind { return Kind::kTypeParam; },
          [](const std::unique_ptr<TypeSpec>& spec) -> Kind {
            return Kind::kType;
          },
          [](const ErrorTypeSpec& spec) -> Kind { return Kind::kError; },
          [](const AbstractType& spec) -> Kind { return Kind::kOpaque; }),
      spec.type_kind());
}

}  // namespace

std::vector<Kind> FunctionDescriptor::FunctionTypeSpecToArgumentKinds(
    const FunctionTypeSpec& spec) {
  std::vector<Kind> kinds;
  kinds.reserve(spec.arg_types().size());
  for (const auto& arg : spec.arg_types()) {
    kinds.push_back(TypeSpecToKind(arg));
  }
  return kinds;
}

}  // namespace cel
