#include "tau/Checkers/Checkers.h"
#include "tau/Core/Analysis.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Basic/FileManager.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>

using namespace clang;
using namespace llvm;
using namespace mlir;

namespace {
cl::OptionCategory TauCategory("Compiler options");
cl::OptionCategory CheckersCategory("Available checkers");

enum class DumpTarget { None, AST, AIR };
cl::opt<DumpTarget>
    DumpAction("dump", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpTarget::AST, "ast", "dump the AST")),
               cl::values(clEnumValN(DumpTarget::AIR, "air", "dump the AIR")),
               cl::cat(TauCategory));

cl::list<std::string> SourcePaths(cl::Positional,
                                  cl::desc("<source-0> [... <source-N>]"),
                                  cl::Required, cl::cat(TauCategory),
                                  cl::sub(*cl::AllSubCommands));

cl::opt<bool> Verify("verify",
                     cl::desc("Verify checker output using comment directives"),
                     cl::cat(TauCategory), cl::Hidden);

} // end anonymous namespace

std::unique_ptr<ScopedDiagnosticHandler>
createHandler(llvm::SourceMgr &SourceManager, mlir::MLIRContext &Context) {
  if (Verify)
    return std::make_unique<SourceMgrDiagnosticVerifierHandler>(
        SourceManager, &Context, llvm::errs());

  return std::make_unique<SourceMgrDiagnosticHandler>(SourceManager, &Context,
                                                      llvm::errs());
}

LogicalResult tauCCMain(int Argc, const char **Argv) {
  tau::chx::registerUseOfUninitChecker();
  tau::core::CheckerCLParser CheckersOptions(CheckersCategory);
  cl::HideUnrelatedOptions({&TauCategory, &CheckersCategory});

  cl::SetVersionPrinter([](raw_ostream &OS) {
    // TODO: remove hardcoded version number
    OS << "tau C/C++ compiler v0.0.1\n";
  });

  auto IR = tau::frontend::runClang(Argc, Argv, SourcePaths);
  if (!IR)
    return failure();

  if (DumpAction == DumpTarget::AIR)
    IR->Module.dump();

  MLIRContext &Context = IR->Context;
  Context.printOpOnDiagnostic(false);

  // TODO: extract all the following logic into a separate component
  PassManager PM(&Context);

  auto ErrorHandler = [&](const Twine &Message) { return failure(); };

  CheckersOptions.addEnabledCheckers(PM);
  PM.addNestedPass<FuncOp>(tau::core::createMainAnalysis());

  if (!Verify) {
    SourceMgrDiagnosticHandler Handler(IR->SourceMgr, &Context, llvm::errs());
    return PM.run(IR->Module);
  }

  SourceMgrDiagnosticVerifierHandler Handler(IR->SourceMgr, &Context,
                                             llvm::errs());
  return success(succeeded(PM.run(IR->Module)) && succeeded(Handler.verify()));
}

int main(int Argc, const char **Argv) { return failed(tauCCMain(Argc, Argv)); }
