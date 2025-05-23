/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_AST_VISITOR_BASE_H_
#define THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_AST_VISITOR_BASE_H_

#include "eval/public/ast_visitor.h"
#include "cel/expr/syntax.pb.h"

namespace google {
namespace api {
namespace expr {
namespace runtime {

// Trivial base implementation of AstVisitor.
class AstVisitorBase : public AstVisitor {
 public:
  AstVisitorBase() {}

  // Non-copyable
  AstVisitorBase(const AstVisitorBase&) = delete;
  AstVisitorBase& operator=(AstVisitorBase const&) = delete;

  ~AstVisitorBase() override {}

  // Const node handler.
  // Invoked after child nodes are processed.
  void PostVisitConst(const cel::expr::Constant*,
                      const cel::expr::Expr*,
                      const SourcePosition*) override {}

  // Ident node handler.
  // Invoked after child nodes are processed.
  void PostVisitIdent(const cel::expr::Expr::Ident*,
                      const cel::expr::Expr*,
                      const SourcePosition*) override {}

  // Select node handler
  // Invoked after child nodes are processed.
  void PostVisitSelect(const cel::expr::Expr::Select*,
                       const cel::expr::Expr*,
                       const SourcePosition*) override {}

  // Call node handler group
  // We provide finer granularity for Call node callbacks to allow special
  // handling for short-circuiting
  // PreVisitCall is invoked before child nodes are processed.
  void PreVisitCall(const cel::expr::Expr::Call*,
                    const cel::expr::Expr*,
                    const SourcePosition*) override {}

  // Invoked after all child nodes are processed.
  void PostVisitCall(const cel::expr::Expr::Call*,
                     const cel::expr::Expr*,
                     const SourcePosition*) override {}

  // Invoked before all child nodes are processed.
  void PreVisitComprehension(const cel::expr::Expr::Comprehension*,
                             const cel::expr::Expr*,
                             const SourcePosition*) override {}

  // Invoked after all child nodes are processed.
  void PostVisitComprehension(const cel::expr::Expr::Comprehension*,
                              const cel::expr::Expr*,
                              const SourcePosition*) override {}

  // Invoked after each argument node processed.
  // For Call arg_num is the index of the argument.
  // For Comprehension arg_num is specified by ComprehensionArg.
  // Expr is the call expression.
  void PostVisitArg(int, const cel::expr::Expr*,
                    const SourcePosition*) override {}

  // Invoked after target node processed.
  void PostVisitTarget(const cel::expr::Expr*,
                       const SourcePosition*) override {}

  // CreateList node handler
  // Invoked after child nodes are processed.
  void PostVisitCreateList(const cel::expr::Expr::CreateList*,
                           const cel::expr::Expr*,
                           const SourcePosition*) override {}

  // CreateStruct node handler
  // Invoked after child nodes are processed.
  void PostVisitCreateStruct(const cel::expr::Expr::CreateStruct*,
                             const cel::expr::Expr*,
                             const SourcePosition*) override {}
};

}  // namespace runtime
}  // namespace expr
}  // namespace api
}  // namespace google

#endif  // THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_AST_VISITOR_BASE_H_
