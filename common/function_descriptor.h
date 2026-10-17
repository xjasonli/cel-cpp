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

#ifndef THIRD_PARTY_CEL_CPP_COMMON_FUNCTION_DESCRIPTOR_H_
#define THIRD_PARTY_CEL_CPP_COMMON_FUNCTION_DESCRIPTOR_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "common/ast.h"
#include "common/kind.h"

namespace cel {

struct FunctionDescriptorOptions {
  // If true (strict, default), error or unknown arguments are propagated
  // instead of calling the function. if false (non-strict), the function may
  // receive error or unknown values as arguments.
  bool is_strict = true;

  // Whether the function is impure or context-sensitive.
  //
  // Impure functions depend on state other than the arguments received during
  // the CEL expression evaluation or have visible side effects. This breaks
  // some of the assumptions of the CEL evaluation model. This flag is used as a
  // hint to the planner that some optimizations are not safe or not effective.
  bool is_contextual = false;
};

// Coarsely describes a function for the purpose of runtime resolution of
// overloads.
class FunctionDescriptor {
 public:
  // Creates a FunctionDescriptor for a global function.
  [[nodiscard]]
  static FunctionDescriptor Global(absl::string_view name,
                                   absl::string_view overload_id,
                                   const FunctionTypeSpec& type,
                                   FunctionDescriptorOptions options = {}) {
    return FunctionDescriptor(name, overload_id, type, options,
                              /*is_receiver_style=*/false);
  }

  // Creates a FunctionDescriptor for a member function.
  [[nodiscard]]
  static FunctionDescriptor Member(absl::string_view name,
                                   absl::string_view overload_id,
                                   const FunctionTypeSpec& type,
                                   FunctionDescriptorOptions options = {}) {
    return FunctionDescriptor(name, overload_id, type, options,
                              /*is_receiver_style=*/true);
  }

  FunctionDescriptor(const FunctionDescriptor&) = default;
  FunctionDescriptor(FunctionDescriptor&&) = default;
  FunctionDescriptor& operator=(const FunctionDescriptor&) = default;
  FunctionDescriptor& operator=(FunctionDescriptor&&) = default;

  FunctionDescriptor(absl::string_view name, bool receiver_style,
                     std::vector<Kind> types, bool is_strict)
      : FunctionDescriptor(name, receiver_style, std::move(types),
                           FunctionDescriptorOptions{is_strict,
                                                     /*is_contextual=*/false}) {
  }

  FunctionDescriptor(absl::string_view name, bool receiver_style,
                     std::vector<Kind> types, bool is_strict,
                     bool is_contextual)
      : FunctionDescriptor(
            name, receiver_style, std::move(types),
            FunctionDescriptorOptions{is_strict, is_contextual}) {}

  FunctionDescriptor(absl::string_view name, bool is_receiver_style,
                     std::vector<Kind> types,
                     FunctionDescriptorOptions options = {})
      : name_(name),
        kinds_(std::move(types)),
        is_receiver_style_(is_receiver_style),
        options_(std::move(options)) {}

  // Function name.
  const std::string& name() const { return name_; }

  // Function overload ID. This may be empty for legacy code.
  [[nodiscard]]
  absl::string_view overload_id() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return overload_id_;
  }

  // Whether function is receiver style i.e. true means arg0.name(args[1:]...).
  bool receiver_style() const { return is_receiver_style_; }

  // The argument types the function accepts.
  //
  // TODO(uncreated-issue/17): make this kinds
  const std::vector<Kind>& types() const { return kinds_; }

  // if true (strict, default), error or unknown arguments are propagated
  // instead of calling the function. if false (non-strict), the function may
  // receive error or unknown values as arguments.
  bool is_strict() const { return options_.is_strict; }

  // Whether the function is contextual (impure).
  //
  // Contextual functions depend on state other than the arguments received in
  // the CEL expression evaluation or have visible side effects. This breaks
  // some of the assumptions of CEL. This flag is used as a hint to the planner
  // that some optimizations are not safe or not effective.
  bool is_contextual() const { return options_.is_contextual; }

  // Helper for matching a descriptor. This tests that the shape is the same --
  // |other| accepts the same number and types of arguments and is the same call
  // style).
  bool ShapeMatches(const FunctionDescriptor& other) const {
    return ShapeMatches(other.receiver_style(), other.types());
  }
  bool ShapeMatches(bool receiver_style, absl::Span<const Kind> types) const;

  bool operator==(const FunctionDescriptor& other) const;

  bool operator<(const FunctionDescriptor& other) const;

 private:
  [[nodiscard]]
  static std::vector<Kind> FunctionTypeSpecToArgumentKinds(
      const FunctionTypeSpec& spec);

  FunctionDescriptor(absl::string_view name, absl::string_view overload_id,
                     const FunctionTypeSpec& type,
                     FunctionDescriptorOptions options, bool is_receiver_style)
      : name_(name),
        overload_id_(overload_id),
        kinds_(FunctionTypeSpecToArgumentKinds(type)),
        is_receiver_style_(is_receiver_style),
        options_(std::move(options)) {}

  std::string name_;
  std::string overload_id_;
  std::vector<Kind> kinds_;
  bool is_receiver_style_;
  FunctionDescriptorOptions options_;
};

}  // namespace cel

#endif  // THIRD_PARTY_CEL_CPP_COMMON_FUNCTION_DESCRIPTOR_H_
