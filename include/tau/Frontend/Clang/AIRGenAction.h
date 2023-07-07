//===- AIRGenAction.h - Clang frontend action for AIR gen -------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/Frontend/Clang/OwningModuleRef.h"
#include "tau/Frontend/Options.h"

#include <mlir/IR/MLIRContext.h>

#include <clang/Tooling/Tooling.h>

namespace tau {
namespace frontend {

class AIRGenAction : public clang::tooling::FrontendActionFactory {
public:
  AIRGenAction(mlir::MLIRContext &Context, Options Opts = {})
      : Context(Context), Opts(Opts) {}
  virtual std::unique_ptr<clang::FrontendAction> create() override;
  mlir::ModuleOp getGeneratedModule() { return Module.release(); }
  mlir::MLIRContext &getContext() { return Context; }

private:
  mlir::MLIRContext &Context;
  Options Opts;
  OwningModuleRef Module;
};

} // end namespace frontend
} // end namespace tau
