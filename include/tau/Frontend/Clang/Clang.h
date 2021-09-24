//===- Clang.h - Run Clang and generate Air ---------------------*- C++ -*-===//
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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>
#include <vector>

namespace clang::tooling {
class CompilationDatabase;
} // end namespace clang::tooling

namespace tau::frontend {

struct Output;

llvm::Expected<std::unique_ptr<clang::tooling::CompilationDatabase>>
readClangOptions(int Argc, const char **Argv);

std::unique_ptr<Output> runClang(const clang::tooling::CompilationDatabase &,
                                 llvm::ArrayRef<std::string> Sources);

std::unique_ptr<Output> runClang(int Argc, const char **Argv,
                                 const llvm::cl::list<std::string> &Sources);

std::unique_ptr<Output>
runClangOnCode(const llvm::Twine &Code, const std::vector<std::string> &Args,
               const llvm::Twine &FileName = "input.cc");

std::unique_ptr<Output>
runClangOnCode(const llvm::Twine &Code,
               const llvm::Twine &FileName = "input.cc");

} // end namespace tau::frontend
