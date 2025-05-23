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
#include "eval/compiler/flat_expr_builder_extensions.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "common/expr.h"
#include "common/native_type.h"
#include "common/value.h"
#include "eval/compiler/resolver.h"
#include "eval/eval/const_value_step.h"
#include "eval/eval/direct_expression_step.h"
#include "eval/eval/evaluator_core.h"
#include "eval/eval/function_step.h"
#include "internal/status_macros.h"
#include "internal/testing.h"
#include "runtime/function_registry.h"
#include "runtime/internal/issue_collector.h"
#include "runtime/internal/runtime_env.h"
#include "runtime/internal/runtime_env_testing.h"
#include "runtime/runtime_issue.h"
#include "runtime/runtime_options.h"
#include "runtime/type_registry.h"
#include "google/protobuf/arena.h"

namespace google::api::expr::runtime {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::cel::Expr;
using ::cel::RuntimeIssue;
using ::cel::runtime_internal::IssueCollector;
using ::cel::runtime_internal::NewTestingRuntimeEnv;
using ::cel::runtime_internal::RuntimeEnv;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;

using Subexpression = ProgramBuilder::Subexpression;

class PlannerContextTest : public testing::Test {
 public:
  PlannerContextTest()
      : env_(NewTestingRuntimeEnv()),
        type_registry_(env_->type_registry),
        function_registry_(env_->function_registry),
        resolver_("", function_registry_, type_registry_,
                  type_registry_.GetComposedTypeProvider()),
        issue_collector_(RuntimeIssue::Severity::kError) {}

 protected:
  ABSL_NONNULL std::shared_ptr<RuntimeEnv> env_;
  cel::TypeRegistry& type_registry_;
  cel::FunctionRegistry& function_registry_;
  cel::RuntimeOptions options_;
  Resolver resolver_;
  IssueCollector issue_collector_;
};

MATCHER_P(UniquePtrHolds, ptr, "") {
  const auto& got = arg;
  return ptr == got.get();
}

struct SimpleTreeSteps {
  const ExpressionStep* a;
  const ExpressionStep* b;
  const ExpressionStep* c;
};

// simulate a program of:
//    a
//   / \
//  b   c
absl::StatusOr<SimpleTreeSteps> InitSimpleTree(
    const Expr& a, const Expr& b, const Expr& c,
    ProgramBuilder& program_builder) {
  CEL_ASSIGN_OR_RETURN(auto a_step, CreateConstValueStep(cel::NullValue(), -1));
  CEL_ASSIGN_OR_RETURN(auto b_step, CreateConstValueStep(cel::NullValue(), -1));
  CEL_ASSIGN_OR_RETURN(auto c_step, CreateConstValueStep(cel::NullValue(), -1));

  SimpleTreeSteps result{a_step.get(), b_step.get(), c_step.get()};

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.AddStep(std::move(b_step));
  program_builder.ExitSubexpression(&b);
  program_builder.EnterSubexpression(&c);
  program_builder.AddStep(std::move(c_step));
  program_builder.ExitSubexpression(&c);
  program_builder.AddStep(std::move(a_step));
  program_builder.ExitSubexpression(&a);

  return result;
}

TEST_F(PlannerContextTest, GetPlan) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto step_ptrs,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_THAT(context.GetSubplan(b), ElementsAre(UniquePtrHolds(step_ptrs.b)));

  EXPECT_THAT(context.GetSubplan(c), ElementsAre(UniquePtrHolds(step_ptrs.c)));

  EXPECT_THAT(context.GetSubplan(a), ElementsAre(UniquePtrHolds(step_ptrs.b),
                                                 UniquePtrHolds(step_ptrs.c),
                                                 UniquePtrHolds(step_ptrs.a)));

  Expr d;
  EXPECT_FALSE(context.IsSubplanInspectable(d));
  EXPECT_THAT(context.GetSubplan(d), IsEmpty());
}

TEST_F(PlannerContextTest, ReplacePlan) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto step_ptrs,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_THAT(context.GetSubplan(a), ElementsAre(UniquePtrHolds(step_ptrs.b),
                                                 UniquePtrHolds(step_ptrs.c),
                                                 UniquePtrHolds(step_ptrs.a)));

  ExecutionPath new_a;

  ASSERT_OK_AND_ASSIGN(auto new_a_step,
                       CreateConstValueStep(cel::NullValue(), -1));
  const ExpressionStep* new_a_step_ptr = new_a_step.get();
  new_a.push_back(std::move(new_a_step));

  ASSERT_THAT(context.ReplaceSubplan(a, std::move(new_a)), IsOk());

  EXPECT_THAT(context.GetSubplan(a),
              ElementsAre(UniquePtrHolds(new_a_step_ptr)));
  EXPECT_THAT(context.GetSubplan(b), IsEmpty());
}

TEST_F(PlannerContextTest, ExtractPlan) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto plan_steps,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_TRUE(context.IsSubplanInspectable(a));
  EXPECT_TRUE(context.IsSubplanInspectable(b));

  ASSERT_OK_AND_ASSIGN(ExecutionPath extracted, context.ExtractSubplan(b));

  EXPECT_THAT(extracted, ElementsAre(UniquePtrHolds(plan_steps.b)));
}

TEST_F(PlannerContextTest, ExtractFailsOnReplacedNode) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_THAT(InitSimpleTree(a, b, c, program_builder).status(), IsOk());

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  ASSERT_THAT(context.ReplaceSubplan(a, {}), IsOk());

  EXPECT_THAT(context.ExtractSubplan(b), IsOkAndHolds(IsEmpty()));
}

TEST_F(PlannerContextTest, ReplacePlanUpdatesParent) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto plan_steps,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_TRUE(context.IsSubplanInspectable(a));

  ASSERT_THAT(context.ReplaceSubplan(c, {}), IsOk());

  EXPECT_THAT(context.GetSubplan(a), ElementsAre(UniquePtrHolds(plan_steps.b),
                                                 UniquePtrHolds(plan_steps.a)));
  EXPECT_THAT(context.GetSubplan(c), IsEmpty());
}

TEST_F(PlannerContextTest, ReplacePlanUpdatesSibling) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto plan_steps,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  ExecutionPath new_b;

  ASSERT_OK_AND_ASSIGN(auto b1_step,
                       CreateConstValueStep(cel::NullValue(), -1));
  const ExpressionStep* b1_step_ptr = b1_step.get();
  new_b.push_back(std::move(b1_step));
  ASSERT_OK_AND_ASSIGN(auto b2_step,
                       CreateConstValueStep(cel::NullValue(), -1));
  const ExpressionStep* b2_step_ptr = b2_step.get();
  new_b.push_back(std::move(b2_step));

  ASSERT_THAT(context.ReplaceSubplan(b, std::move(new_b)), IsOk());

  EXPECT_THAT(context.GetSubplan(c), ElementsAre(UniquePtrHolds(plan_steps.c)));
  EXPECT_THAT(context.GetSubplan(b), ElementsAre(UniquePtrHolds(b1_step_ptr),
                                                 UniquePtrHolds(b2_step_ptr)));
  EXPECT_THAT(
      context.GetSubplan(a),
      ElementsAre(UniquePtrHolds(b1_step_ptr), UniquePtrHolds(b2_step_ptr),
                  UniquePtrHolds(plan_steps.c), UniquePtrHolds(plan_steps.a)));
}

TEST_F(PlannerContextTest, ReplacePlanFailsOnUpdatedNode) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto plan_steps,
                       InitSimpleTree(a, b, c, program_builder));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_THAT(context.GetSubplan(a), ElementsAre(UniquePtrHolds(plan_steps.b),
                                                 UniquePtrHolds(plan_steps.c),
                                                 UniquePtrHolds(plan_steps.a)));

  ASSERT_THAT(context.ReplaceSubplan(a, {}), IsOk());
  EXPECT_THAT(context.ReplaceSubplan(b, {}), IsOk());
}

TEST_F(PlannerContextTest, AddSubplanStep) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(auto plan_steps,
                       InitSimpleTree(a, b, c, program_builder));

  ASSERT_OK_AND_ASSIGN(auto b2_step,
                       CreateConstValueStep(cel::NullValue(), -1));

  const ExpressionStep* b2_step_ptr = b2_step.get();

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  ASSERT_THAT(context.AddSubplanStep(b, std::move(b2_step)), IsOk());

  EXPECT_THAT(context.GetSubplan(b), ElementsAre(UniquePtrHolds(plan_steps.b),
                                                 UniquePtrHolds(b2_step_ptr)));
  EXPECT_THAT(context.GetSubplan(c), ElementsAre(UniquePtrHolds(plan_steps.c)));
  EXPECT_THAT(
      context.GetSubplan(a),
      ElementsAre(UniquePtrHolds(plan_steps.b), UniquePtrHolds(b2_step_ptr),
                  UniquePtrHolds(plan_steps.c), UniquePtrHolds(plan_steps.a)));
}

TEST_F(PlannerContextTest, AddSubplanStepFailsOnUnknownNode) {
  Expr a;
  Expr b;
  Expr c;
  Expr d;
  ProgramBuilder program_builder;

  ASSERT_THAT(InitSimpleTree(a, b, c, program_builder).status(), IsOk());

  ASSERT_OK_AND_ASSIGN(auto b2_step,
                       CreateConstValueStep(cel::NullValue(), -1));

  std::shared_ptr<google::protobuf::Arena> arena;
  PlannerContext context(env_, resolver_, options_,
                         type_registry_.GetComposedTypeProvider(),
                         issue_collector_, program_builder, arena);

  EXPECT_THAT(context.GetSubplan(d), IsEmpty());

  EXPECT_THAT(context.AddSubplanStep(d, std::move(b2_step)),
              StatusIs(absl::StatusCode::kInternal));
}

class ProgramBuilderTest : public testing::Test {
 public:
  ProgramBuilderTest() : type_registry_(), function_registry_() {}

 protected:
  cel::TypeRegistry type_registry_;
  cel::FunctionRegistry function_registry_;
};

TEST_F(ProgramBuilderTest, ExtractSubexpression) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(SimpleTreeSteps step_ptrs,
                       InitSimpleTree(a, b, c, program_builder));
  EXPECT_EQ(program_builder.ExtractSubexpression(&c), 0);
  EXPECT_EQ(program_builder.ExtractSubexpression(&b), 1);

  EXPECT_THAT(program_builder.FlattenMain(),
              ElementsAre(UniquePtrHolds(step_ptrs.a)));
  EXPECT_THAT(program_builder.FlattenSubexpressions(),
              ElementsAre(ElementsAre(UniquePtrHolds(step_ptrs.c)),
                          ElementsAre(UniquePtrHolds(step_ptrs.b))));
}

TEST_F(ProgramBuilderTest, FlattenRemovesChildrenReferences) {
  Expr a;
  Expr b;
  Expr c;
  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.EnterSubexpression(&c);
  program_builder.ExitSubexpression(&c);
  program_builder.ExitSubexpression(&b);
  program_builder.ExitSubexpression(&a);

  auto subexpr_b = program_builder.GetSubexpression(&b);
  ASSERT_TRUE(subexpr_b != nullptr);
  subexpr_b->Flatten();

  auto* subexpr_c = program_builder.GetSubexpression(&c);
  EXPECT_EQ(subexpr_b->ExtractChild(subexpr_c), nullptr);
}

TEST_F(ProgramBuilderTest, ExtractReturnsNullOnFlattendExpr) {
  Expr a;
  Expr b;
  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.ExitSubexpression(&b);
  program_builder.ExitSubexpression(&a);

  auto* subexpr_a = program_builder.GetSubexpression(&a);
  auto* subexpr_b = program_builder.GetSubexpression(&b);

  ASSERT_TRUE(subexpr_a != nullptr);
  ASSERT_TRUE(subexpr_b != nullptr);

  subexpr_a->Flatten();
  // subexpr_b is now freed.

  EXPECT_EQ(subexpr_a->ExtractChild(subexpr_b), nullptr);
  EXPECT_EQ(program_builder.ExtractSubexpression(&b), -1);
}

TEST_F(ProgramBuilderTest, ExtractReturnsNullOnNonChildren) {
  Expr a;
  Expr b;
  Expr c;

  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.EnterSubexpression(&c);
  program_builder.ExitSubexpression(&c);
  program_builder.ExitSubexpression(&b);
  program_builder.ExitSubexpression(&a);

  auto* subexpr_a = program_builder.GetSubexpression(&a);
  auto* subexpr_c = program_builder.GetSubexpression(&c);

  ASSERT_TRUE(subexpr_a != nullptr);
  ASSERT_TRUE(subexpr_c != nullptr);

  EXPECT_EQ(subexpr_a->ExtractChild(subexpr_c), nullptr);
}

TEST_F(ProgramBuilderTest, ResetWorks) {
  Expr a;
  Expr b;
  Expr c;

  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.EnterSubexpression(&c);
  program_builder.ExitSubexpression(&c);
  program_builder.ExitSubexpression(&b);
  program_builder.ExitSubexpression(&a);

  auto* subexpr_a = program_builder.GetSubexpression(&a);
  auto* subexpr_c = program_builder.GetSubexpression(&c);

  ASSERT_TRUE(subexpr_a != nullptr);
  ASSERT_TRUE(subexpr_c != nullptr);

  program_builder.Reset();

  subexpr_a = program_builder.GetSubexpression(&a);
  subexpr_c = program_builder.GetSubexpression(&c);

  ASSERT_TRUE(subexpr_a == nullptr);
  ASSERT_TRUE(subexpr_c == nullptr);
}

TEST_F(ProgramBuilderTest, ExtractWorks) {
  Expr a;
  Expr b;
  Expr c;

  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.ExitSubexpression(&b);

  ASSERT_OK_AND_ASSIGN(auto a_step, CreateConstValueStep(cel::NullValue(), -1));
  program_builder.AddStep(std::move(a_step));
  program_builder.EnterSubexpression(&c);
  program_builder.ExitSubexpression(&c);
  program_builder.ExitSubexpression(&a);

  auto* subexpr_a = program_builder.GetSubexpression(&a);
  auto* subexpr_c = program_builder.GetSubexpression(&c);

  ASSERT_TRUE(subexpr_a != nullptr);
  ASSERT_TRUE(subexpr_c != nullptr);

  EXPECT_EQ(subexpr_a->ExtractChild(subexpr_c), subexpr_c);
}

TEST_F(ProgramBuilderTest, ExtractToRequiresFlatten) {
  Expr a;
  Expr b;
  Expr c;

  ProgramBuilder program_builder;

  ASSERT_OK_AND_ASSIGN(SimpleTreeSteps step_ptrs,
                       InitSimpleTree(a, b, c, program_builder));

  auto* subexpr_a = program_builder.GetSubexpression(&a);
  ExecutionPath path;

  EXPECT_FALSE(subexpr_a->ExtractTo(path));

  subexpr_a->Flatten();
  EXPECT_TRUE(subexpr_a->ExtractTo(path));

  EXPECT_THAT(path, ElementsAre(UniquePtrHolds(step_ptrs.b),
                                UniquePtrHolds(step_ptrs.c),
                                UniquePtrHolds(step_ptrs.a)));
}

TEST_F(ProgramBuilderTest, Recursive) {
  Expr a;
  Expr b;
  Expr c;

  ProgramBuilder program_builder;

  program_builder.EnterSubexpression(&a);
  program_builder.EnterSubexpression(&b);
  program_builder.current()->set_recursive_program(
      CreateConstValueDirectStep(cel::NullValue()), 1);
  program_builder.ExitSubexpression(&b);
  program_builder.EnterSubexpression(&c);
  program_builder.current()->set_recursive_program(
      CreateConstValueDirectStep(cel::NullValue()), 1);
  program_builder.ExitSubexpression(&c);

  ASSERT_FALSE(program_builder.current()->IsFlattened());
  ASSERT_FALSE(program_builder.current()->IsRecursive());
  ASSERT_TRUE(program_builder.GetSubexpression(&b)->IsRecursive());
  ASSERT_TRUE(program_builder.GetSubexpression(&c)->IsRecursive());

  EXPECT_EQ(program_builder.GetSubexpression(&b)->recursive_program().depth, 1);
  EXPECT_EQ(program_builder.GetSubexpression(&c)->recursive_program().depth, 1);

  cel::CallExpr call_expr;
  call_expr.set_function("_==_");
  call_expr.mutable_args().emplace_back();
  call_expr.mutable_args().emplace_back();

  auto max_depth = program_builder.current()->RecursiveDependencyDepth();

  EXPECT_THAT(max_depth, Optional(1));

  auto deps = program_builder.current()->ExtractRecursiveDependencies();

  program_builder.current()->set_recursive_program(
      CreateDirectFunctionStep(-1, call_expr, std::move(deps), {}),
      *max_depth + 1);

  program_builder.ExitSubexpression(&a);

  auto path = program_builder.FlattenMain();

  ASSERT_THAT(path, testing::SizeIs(1));
  EXPECT_TRUE(path[0]->GetNativeTypeId() ==
              cel::NativeTypeId::For<WrappedDirectStep>());
}

}  // namespace
}  // namespace google::api::expr::runtime
