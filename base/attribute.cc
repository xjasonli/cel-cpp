// Copyright 2022 Google LLC
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

#include "base/attribute.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/compare.h"
#include "base/kind.h"
#include "internal/lexis.h"
#include "internal/status_macros.h"
#include "internal/strings.h"

namespace cel {

namespace {

// Visitor for appending string representation for different qualifier kinds.
class AttributeStringPrinter {
 public:
  // String representation for the given qualifier is appended to output.
  // output must be non-null.
  explicit AttributeStringPrinter(std::string* output) : output_(*output) {}

  absl::Status operator()(std::monostate) const {
    return absl::InvalidArgumentError("bad attribute qualifier");
  }

  absl::Status operator()(int64_t index) {
    absl::StrAppend(&output_, "[", index, "]");
    return absl::OkStatus();
  }

  absl::Status operator()(uint64_t index) {
    absl::StrAppend(&output_, "[", index, "]");
    return absl::OkStatus();
  }

  absl::Status operator()(bool bool_key) {
    absl::StrAppend(&output_, "[", (bool_key) ? "true" : "false", "]");
    return absl::OkStatus();
  }

  absl::Status operator()(const std::string& field) {
    absl::StrAppend(&output_, ".", field);
    return absl::OkStatus();
  }

 private:
  std::string& output_;
};

// Visitor for appending string representation for different qualifier kinds.
class AttributeQualifierStringPrinter {
 public:
  // String representation for the given qualifier is appended to output.
  explicit AttributeQualifierStringPrinter(std::string* absl_nonnull output)
      : output_(*output) {}

  absl::Status operator()(std::monostate) const {
    // Attributes are represented as a variant, with illegal attribute
    // qualifiers represented with their type as the first alternative.
    return absl::InvalidArgumentError("bad attribute qualifier");
  }

  absl::Status operator()(int64_t index) {
    absl::StrAppend(&output_, index);
    return absl::OkStatus();
  }

  absl::Status operator()(uint64_t index) {
    absl::StrAppend(&output_, index);
    return absl::OkStatus();
  }

  absl::Status operator()(bool bool_key) {
    absl::StrAppend(&output_, (bool_key) ? "true" : "false");
    return absl::OkStatus();
  }

  absl::Status operator()(const std::string& field) {
    absl::StrAppend(&output_, field);
    return absl::OkStatus();
  }

 private:
  std::string& output_;
};

struct AttributeQualifierTypeVisitor final {
  Kind operator()(std::monostate) const { return Kind::kNull; }

  Kind operator()(int64_t) const { return Kind::kInt64; }

  Kind operator()(uint64_t) const { return Kind::kUint64; }

  Kind operator()(const std::string&) const { return Kind::kString; }

  Kind operator()(bool) const { return Kind::kBool; }
};

struct AttributeQualifierToString {
  void operator()(std::monostate) const {}

  void operator()(bool value) const { output.append(value ? "true" : "false"); }

  void operator()(int64_t value) const { absl::StrAppend(&output, value); }

  void operator()(uint64_t value) const {
    absl::StrAppend(&output, value, "u");
  }

  void operator()(absl::string_view value) const {
    if (internal::LexisIsIdentifier(value)) {
      absl::StrAppend(&output, value);
    } else {
      internal::FormatStringLiteralTo(value, &output);
    }
  }

  void operator()(common_internal::WildcardType) const {
    output.push_back('*');
  }

  std::string& output;
};

struct AttributeQualifierJoinString {
  void operator()(std::monostate) const {}

  void operator()(bool value) const {
    output.push_back('[');
    output.append(value ? "true" : "false");
    output.push_back(']');
  }

  void operator()(int64_t value) const {
    absl::StrAppend(&output, "[", value, "]");
  }

  void operator()(uint64_t value) const {
    absl::StrAppend(&output, "[", value, "u", "]");
  }

  void operator()(absl::string_view value) const {
    if (internal::LexisIsIdentifier(value)) {
      absl::StrAppend(&output, ".", value);
    } else {
      output.push_back('[');
      internal::FormatStringLiteralTo(value, &output);
      output.push_back(']');
    }
  }

  void operator()(common_internal::WildcardType) const { output.append(".*"); }

  std::string& output;
};

template <typename Q>
void AttributeQualifierAppendJoinString(std::string* out,
                                        absl::Span<const Q> qualifiers) {
  for (const auto& qualifier : qualifiers) {
    std::visit(AttributeQualifierJoinString{*out},
               common_internal::AsVariant(qualifier));
  }
}

}  // namespace

Kind AttributeQualifier::kind() const {
  return std::visit(AttributeQualifierTypeVisitor{}, value_);
}

absl::StatusOr<std::string> Attribute::AsString() const {
  if (variable_name().empty()) {
    return absl::InvalidArgumentError(
        "Only ident rooted attributes are supported.");
  }

  std::string result = std::string(variable_name());

  for (const auto& qualifier : qualifier_path()) {
    CEL_RETURN_IF_ERROR(std::visit(AttributeStringPrinter(&result),
                                   common_internal::AsVariant(qualifier)));
  }

  return result;
}

std::string Attribute::ToString() const {
  std::string result;
  result.append(variable_name());
  AttributeQualifier::AppendJoinString(&result, qualifier_path());
  return result;
}

std::string AttributePattern::ToString() const {
  std::string result;
  result.append(variable());
  AttributeQualifierPattern::AppendJoinString(&result, qualifier_path());
  return result;
}

void AttributeQualifier::AppendToString(std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierToString{*out}, value_);
}

void AttributeQualifierView::AppendToString(
    std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierToString{*out}, value_);
}

void AttributeQualifierPattern::AppendToString(
    std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierToString{*out}, value_);
}

void AttributeQualifier::AppendJoinString(std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierJoinString{*out}, value_);
}

void AttributeQualifierView::AppendJoinString(
    std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierJoinString{*out}, value_);
}

void AttributeQualifierPattern::AppendJoinString(
    std::string* absl_nonnull out) const {
  std::visit(AttributeQualifierJoinString{*out}, value_);
}

void AttributeQualifier::AppendJoinString(
    std::string* absl_nonnull out,
    absl::Span<const AttributeQualifier> qualifiers) {
  AttributeQualifierAppendJoinString(out, qualifiers);
}

void AttributeQualifierView::AppendJoinString(
    std::string* absl_nonnull out,
    absl::Span<const AttributeQualifierView> qualifiers) {
  AttributeQualifierAppendJoinString(out, qualifiers);
}

void AttributeQualifierPattern::AppendJoinString(
    std::string* absl_nonnull out,
    absl::Span<const AttributeQualifierPattern> qualifiers) {
  AttributeQualifierAppendJoinString(out, qualifiers);
}

bool AttributeQualifier::IsMatch(const AttributeQualifier& other) const {
  ABSL_DCHECK(*this) << "bad attribute qualifier";
  ABSL_DCHECK(other) << "bad attribute qualifier";
  if (std::holds_alternative<std::monostate>(value_) ||
      std::holds_alternative<std::monostate>(other.value_)) {
    return false;
  }
  return *this == other;
}

bool AttributeQualifier::IsMatch(absl::string_view other_key) const {
  ABSL_DCHECK(*this) << "bad attribute qualifier";
  if (auto string = AsString(); string.has_value()) {
    return *string == other_key;
  }
  return false;
}

bool AttributeQualifierPattern::IsMatch(
    const AttributeQualifier& qualifier) const {
  ABSL_DCHECK(*this) << "bad attribute qualifier pattern";
  ABSL_DCHECK(qualifier) << "bad attribute qualifier";
  if (!*this || !qualifier) {
    return false;
  }
  if (IsWildcard()) {
    return true;
  }
  return *this == qualifier;
}

bool AttributeQualifierPattern::IsMatch(absl::string_view other_key) const {
  ABSL_DCHECK(*this) << "bad attribute qualifier pattern";
  if (IsWildcard()) {
    return true;
  }
  if (auto string = AsString(); string.has_value()) {
    return *string == other_key;
  }
  return false;
}

namespace {

struct AttributeQualifierEqualTo {
  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(const std::monostate&,
                                               const std::monostate&) const {
    return false;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>, bool>
  operator()(const std::monostate&, const T&) const {
    return false;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>, bool>
  operator()(const T&, const std::monostate&) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(bool lhs, bool rhs) const {
    return lhs == rhs;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(bool, int64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(bool, uint64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(bool, absl::string_view) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(int64_t, bool) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(int64_t lhs, int64_t rhs) const {
    return lhs == rhs;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(int64_t, uint64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(int64_t,
                                               absl::string_view) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(uint64_t, bool) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(uint64_t, int64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(uint64_t lhs,
                                               uint64_t rhs) const {
    return lhs == rhs;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(uint64_t,
                                               absl::string_view) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(absl::string_view, bool) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(absl::string_view,
                                               int64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(absl::string_view,
                                               uint64_t) const {
    return false;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(absl::string_view lhs,
                                               absl::string_view rhs) const {
    return lhs == rhs;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator()(
      const common_internal::WildcardType&,
      const common_internal::WildcardType&) const {
    return true;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, common_internal::WildcardType> &&
          !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
      bool>
  operator()(const common_internal::WildcardType&, const T&) const {
    return false;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, common_internal::WildcardType> &&
          !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
      bool>
  operator()(const T&, const common_internal::WildcardType&) const {
    return false;
  }
};

template <typename T>
[[nodiscard]]
ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering SynthesizeCompare(T lhs,
                                                                     T rhs) {
#if defined(__cpp_impl_three_way_comparison) && \
    defined(__cpp_lib_three_way_comparison)
  return lhs <=> rhs;
#else
  if (lhs < rhs) {
    return absl::strong_ordering::less;
  }
  if (rhs < lhs) {
    return absl::strong_ordering::greater;
  }
  return absl::strong_ordering::equal;
#endif
}

struct AttributeQualifierCompare {
  // monostate < int < uint < string < bool < wildcard

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      const std::monostate&, const std::monostate&) const {
    return absl::strong_ordering::equal;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<!std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
                       absl::strong_ordering>
      operator()(const std::monostate&, const T&) const {
    return absl::strong_ordering::less;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<!std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
                       absl::strong_ordering>
      operator()(const T&, const std::monostate&) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      bool lhs, bool rhs) const {
    return (SynthesizeCompare)(lhs, rhs);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(bool,
                                                                int64_t) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      bool, uint64_t) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      bool, absl::string_view) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(int64_t,
                                                                bool) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      int64_t lhs, int64_t rhs) const {
    return (SynthesizeCompare)(lhs, rhs);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      int64_t, uint64_t) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      int64_t, absl::string_view) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(uint64_t,
                                                                bool) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(uint64_t,
                                                                int64_t) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      uint64_t lhs, uint64_t rhs) const {
    return (SynthesizeCompare)(lhs, rhs);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      uint64_t, absl::string_view) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      absl::string_view, bool) const {
    return absl::strong_ordering::less;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      absl::string_view, int64_t) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      absl::string_view, uint64_t) const {
    return absl::strong_ordering::greater;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      absl::string_view lhs, absl::string_view rhs) const {
    return (SynthesizeCompare)(lhs, rhs);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE absl::strong_ordering operator()(
      const common_internal::WildcardType&,
      const common_internal::WildcardType&) const {
    return absl::strong_ordering::equal;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, common_internal::WildcardType> &&
          !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
      absl::strong_ordering>
  operator()(const common_internal::WildcardType&, const T&) const {
    return absl::strong_ordering::greater;
  }

  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<
      !std::is_same_v<absl::remove_cvref_t<T>, common_internal::WildcardType> &&
          !std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
      absl::strong_ordering>
  operator()(const T&, const common_internal::WildcardType&) const {
    return absl::strong_ordering::less;
  }
};

}  // namespace

absl::strong_ordering AttributeQualifier::Compare(
    const AttributeQualifier& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifier::Compare(
    const AttributeQualifierView& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifier::Compare(
    const AttributeQualifierPattern& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierView::Compare(
    const AttributeQualifier& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierView::Compare(
    const AttributeQualifierView& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierView::Compare(
    const AttributeQualifierPattern& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierPattern::Compare(
    const AttributeQualifier& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierPattern::Compare(
    const AttributeQualifierView& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering AttributeQualifierPattern::Compare(
    const AttributeQualifierPattern& other) const {
  return std::visit(AttributeQualifierCompare{},
                    common_internal::AsVariant(*this),
                    common_internal::AsVariant(other));
}

absl::strong_ordering Attribute::Compare(const Attribute& other) const {
  if (absl::strong_ordering order =
          (SynthesizeCompare)(variable_name(), other.variable_name());
      order != absl::strong_ordering::equal) {
    return order;
  }
  const auto& qualifiers = qualifier_path();
  const auto& other_qualifiers = other.qualifier_path();
  const size_t min_size = std::min(qualifiers.size(), other_qualifiers.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (absl::strong_ordering order =
            qualifiers[i].Compare(other_qualifiers[i]);
        order != absl::strong_ordering::equal) {
      return order;
    }
  }
  if (qualifiers.size() < other_qualifiers.size()) {
    return absl::strong_ordering::less;
  }
  if (qualifiers.size() > other_qualifiers.size()) {
    return absl::strong_ordering::greater;
  }
  return absl::strong_ordering::equal;
}

bool operator==(const AttributeQualifier& lhs, const AttributeQualifier& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierView& lhs,
                const AttributeQualifierView& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierPattern& lhs,
                const AttributeQualifierPattern& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifier& lhs,
                const AttributeQualifierView& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifier& lhs,
                const AttributeQualifierPattern& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierView& lhs,
                const AttributeQualifier& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierView& lhs,
                const AttributeQualifierPattern& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierPattern& lhs,
                const AttributeQualifier& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const AttributeQualifierPattern& lhs,
                const AttributeQualifierView& rhs) {
  return std::visit(AttributeQualifierEqualTo{},
                    common_internal::AsVariant(lhs),
                    common_internal::AsVariant(rhs));
}

bool operator==(const Attribute& lhs, const Attribute& rhs) {
  return lhs.variable_name() == rhs.variable_name() &&
         absl::c_equal(lhs.qualifier_path(), rhs.qualifier_path());
}

bool operator==(const AttributePattern& lhs, const AttributePattern& rhs) {
  return lhs.variable() == rhs.variable() &&
         absl::c_equal(lhs.qualifier_path(), rhs.qualifier_path());
}

}  // namespace cel
