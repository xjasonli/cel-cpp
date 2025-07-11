#include "eval/eval/attribute_utility.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "base/attribute.h"
#include "base/attribute_set.h"
#include "base/function_result.h"
#include "base/function_result_set.h"
#include "base/internal/unknown_set.h"
#include "common/casting.h"
#include "common/function_descriptor.h"
#include "common/unknown.h"
#include "common/value.h"
#include "eval/eval/attribute_trail.h"
#include "eval/internal/errors.h"
#include "internal/status_macros.h"
#include "runtime/internal/attribute_matcher.h"

namespace google::api::expr::runtime {

using ::cel::Attribute;
using ::cel::AttributePattern;
using ::cel::AttributeSet;
using ::cel::Cast;
using ::cel::ErrorValue;
using ::cel::FunctionResult;
using ::cel::FunctionResultSet;
using ::cel::InstanceOf;
using ::cel::UnknownValue;
using ::cel::Value;
using ::cel::base_internal::UnknownSet;
using ::cel::runtime_internal::AttributeMatcher;

using Accumulator = AttributeUtility::Accumulator;
using MatchResult = AttributeMatcher::MatchResult;

DefaultAttributeMatcher::DefaultAttributeMatcher(
    absl::Span<const AttributePattern> unknown_patterns,
    absl::Span<const AttributePattern> missing_patterns)
    : unknown_patterns_(unknown_patterns),
      missing_patterns_(missing_patterns) {}

DefaultAttributeMatcher::DefaultAttributeMatcher() = default;

AttributeMatcher::MatchResult MatchAgainstPatterns(
    absl::Span<const AttributePattern> patterns, const Attribute& attr) {
  MatchResult result = MatchResult::NONE;
  for (const auto& pattern : patterns) {
    auto current_match = pattern.IsMatch(attr);
    if (current_match == cel::AttributePattern::MatchType::FULL) {
      return MatchResult::FULL;
    }
    if (current_match == cel::AttributePattern::MatchType::PARTIAL) {
      result = MatchResult::PARTIAL;
    }
  }
  return result;
}

DefaultAttributeMatcher::MatchResult DefaultAttributeMatcher::CheckForUnknown(
    const Attribute& attr) const {
  return MatchAgainstPatterns(unknown_patterns_, attr);
}

DefaultAttributeMatcher::MatchResult DefaultAttributeMatcher::CheckForMissing(
    const Attribute& attr) const {
  return MatchAgainstPatterns(missing_patterns_, attr);
}

bool AttributeUtility::CheckForMissingAttribute(
    const AttributeTrail& trail) const {
  if (trail.empty()) {
    return false;
  }
  // Missing attributes are only treated as errors if the attribute exactly
  // matches (so no guard against passing partial state to a function as with
  // unknowns). This was initially a design oversight, but is difficult to
  // change now.
  return matcher_->CheckForMissing(trail.attribute()) ==
         AttributeMatcher::MatchResult::FULL;
}

// Checks whether particular corresponds to any patterns that define unknowns.
bool AttributeUtility::CheckForUnknown(const AttributeTrail& trail,
                                       bool use_partial) const {
  if (trail.empty()) {
    return false;
  }
  MatchResult result = matcher_->CheckForUnknown(trail.attribute());

  if (result == MatchResult::FULL ||
      (use_partial && result == MatchResult::PARTIAL)) {
    return true;
  }
  return false;
}

// Creates merged UnknownAttributeSet.
// Scans over the args collection, merges any UnknownSets found in
// it together with initial_set (if initial_set is not null).
// Returns pointer to merged set or nullptr, if there were no sets to merge.
absl::optional<UnknownValue> AttributeUtility::MergeUnknowns(
    absl::Span<const cel::Value> args) const {
  // Empty unknown value may be used as a sentinel in some tests so need to
  // distinguish unset (nullopt) and empty(engaged empty value).
  absl::optional<UnknownSet> result_set;

  for (const auto& value : args) {
    if (!value->Is<cel::UnknownValue>()) continue;
    if (!result_set.has_value()) {
      result_set.emplace();
    }
    const auto& current_set = value.GetUnknown();

    cel::base_internal::UnknownSetAccess::Add(
        *result_set, UnknownSet(current_set.attribute_set(),
                                current_set.function_result_set()));
  }

  if (!result_set.has_value()) {
    return absl::nullopt;
  }

  return UnknownValue(cel::Unknown(result_set->unknown_attributes(),
                                   result_set->unknown_function_results()));
}

UnknownValue AttributeUtility::MergeUnknownValues(
    const UnknownValue& left, const UnknownValue& right) const {
  // Empty unknown value may be used as a sentinel in some tests so need to
  // distinguish unset (nullopt) and empty(engaged empty value).
  AttributeSet attributes;
  FunctionResultSet function_results;
  attributes.Add(left.attribute_set());
  function_results.Add(left.function_result_set());
  attributes.Add(right.attribute_set());
  function_results.Add(right.function_result_set());

  return UnknownValue(
      cel::Unknown(std::move(attributes), std::move(function_results)));
}

// Creates merged UnknownAttributeSet.
// Scans over the args collection, determines if there matches to unknown
// patterns, merges attributes together with those from initial_set
// (if initial_set is not null).
// Returns pointer to merged set or nullptr, if there were no sets to merge.
AttributeSet AttributeUtility::CheckForUnknowns(
    absl::Span<const AttributeTrail> args, bool use_partial) const {
  AttributeSet attribute_set;

  for (const auto& trail : args) {
    if (CheckForUnknown(trail, use_partial)) {
      attribute_set.Add(trail.attribute());
    }
  }

  return attribute_set;
}

// Creates merged UnknownAttributeSet.
// Merges together attributes from UnknownAttributeSets found in the args
// collection, attributes from attr that match unknown pattern
// patterns, and attributes from initial_set
// (if initial_set is not null).
// Returns pointer to merged set or nullptr, if there were no sets to merge.
absl::optional<UnknownValue> AttributeUtility::IdentifyAndMergeUnknowns(
    absl::Span<const cel::Value> args, absl::Span<const AttributeTrail> attrs,
    bool use_partial) const {
  absl::optional<UnknownSet> result_set;

  // Identify new unknowns by attribute patterns.
  cel::AttributeSet attr_set = CheckForUnknowns(attrs, use_partial);
  if (!attr_set.empty()) {
    result_set.emplace(std::move(attr_set));
  }

  // merge down existing unknown sets
  absl::optional<UnknownValue> arg_unknowns = MergeUnknowns(args);

  if (!result_set.has_value()) {
    // No new unknowns so no need to check for presence of existing unknowns --
    // just forward.
    return arg_unknowns;
  }

  if (arg_unknowns.has_value()) {
    cel::base_internal::UnknownSetAccess::Add(
        *result_set, UnknownSet((*arg_unknowns).attribute_set(),
                                (*arg_unknowns).function_result_set()));
  }

  return UnknownValue(cel::Unknown(result_set->unknown_attributes(),
                                   result_set->unknown_function_results()));
}

UnknownValue AttributeUtility::CreateUnknownSet(cel::Attribute attr) const {
  return UnknownValue(cel::Unknown(AttributeSet({std::move(attr)})));
}

absl::StatusOr<ErrorValue> AttributeUtility::CreateMissingAttributeError(
    const cel::Attribute& attr) const {
  CEL_ASSIGN_OR_RETURN(std::string message, attr.AsString());
  return cel::ErrorValue(
      cel::runtime_internal::CreateMissingAttributeError(message));
}

UnknownValue AttributeUtility::CreateUnknownSet(
    const cel::FunctionDescriptor& fn_descriptor, int64_t expr_id,
    absl::Span<const cel::Value> args) const {
  return UnknownValue(
      cel::Unknown(FunctionResultSet(FunctionResult(fn_descriptor, expr_id))));
}

void AttributeUtility::Add(Accumulator& a, const cel::UnknownValue& v) const {
  a.attribute_set_.Add(v.attribute_set());
  a.function_result_set_.Add(v.function_result_set());
}

void AttributeUtility::Add(Accumulator& a, const AttributeTrail& attr) const {
  a.attribute_set_.Add(attr.attribute());
}

void Accumulator::Add(const UnknownValue& value) {
  unknown_present_ = true;
  parent_.Add(*this, value);
}

void Accumulator::Add(const AttributeTrail& attr) { parent_.Add(*this, attr); }

void Accumulator::MaybeAdd(const Value& v) {
  if (v.IsUnknown()) {
    Add(v.GetUnknown());
  }
}

void Accumulator::MaybeAdd(const Value& v, const AttributeTrail& attr) {
  if (v.IsUnknown()) {
    Add(v.GetUnknown());
  } else if (parent_.CheckForUnknown(attr, /*use_partial=*/true)) {
    Add(attr);
  }
}

bool Accumulator::IsEmpty() const {
  return !unknown_present_ && attribute_set_.empty() &&
         function_result_set_.empty();
}

cel::UnknownValue Accumulator::Build() && {
  return cel::UnknownValue(
      cel::Unknown(std::move(attribute_set_), std::move(function_result_set_)));
}

}  // namespace google::api::expr::runtime
