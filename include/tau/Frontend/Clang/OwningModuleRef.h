//===- OwningModuleRef.h - A simple typedef for owning module ---*- C++ -*-===//
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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

namespace tau::frontend {
using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;
} // end namespace tau::frontend
