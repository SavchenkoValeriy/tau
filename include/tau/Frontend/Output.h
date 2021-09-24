//===- Output.h - Frontend results ------------------------------*- C++ -*-===//
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

#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>

namespace tau::frontend {

struct Output {
  llvm::SourceMgr SourceMgr;
  mlir::ModuleOp Module;
  mlir::MLIRContext Context;
};

} // end namespace tau::frontend
