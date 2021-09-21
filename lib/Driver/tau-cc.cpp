#include "tau/Checkers/Checkers.h"
#include "tau/Core/Analysis.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

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

Expected<std::unique_ptr<tooling::CompilationDatabase>>
readClangOptions(int Argc, const char **Argv) {
  std::string ErrorMessage;
  auto Result = tooling::FixedCompilationDatabase::loadFromCommandLine(
      Argc, Argv, ErrorMessage);
  if (!ErrorMessage.empty())
    ErrorMessage.append("\n");
  llvm::raw_string_ostream OS(ErrorMessage);
  // Stop initializing if command-line option parsing failed.
  if (!cl::ParseCommandLineOptions(Argc, Argv, "", &OS)) {
    OS.flush();
    return llvm::make_error<llvm::StringError>(ErrorMessage,
                                               llvm::inconvertibleErrorCode());
  }

  if (!Result)
    Result.reset(new tooling::FixedCompilationDatabase(".", {}));

  return Result;
}

std::unique_ptr<ScopedDiagnosticHandler>
createHandler(llvm::SourceMgr &SourceManager, mlir::MLIRContext &Context) {
  if (Verify)
    return std::make_unique<SourceMgrDiagnosticVerifierHandler>(
        SourceManager, &Context, llvm::errs());

  return std::make_unique<SourceMgrDiagnosticHandler>(SourceManager, &Context,
                                                      llvm::errs());
}

struct FrontendOutput {
  llvm::SourceMgr SourceMgr;
  mlir::ModuleOp Module;
  mlir::MLIRContext Context;
};

std::unique_ptr<FrontendOutput> runClang(int Argc, const char **Argv) {
  auto CDB = readClangOptions(Argc, Argv);
  if (auto E = CDB.takeError()) {
    llvm::errs() << toString(std::move(E));
    return nullptr;
  }

  auto Result = std::make_unique<FrontendOutput>();

  tooling::ClangTool Tool(*CDB->get(), SourcePaths);
  tau::frontend::AIRGenAction Generator{Result->Context};
  if (Tool.run(&Generator))
    // TODO: output clang errors?
    return nullptr;

  clang::FileManager &FileMgr = Tool.getFiles();
  for (StringRef SourcePath : Tool.getSourcePaths()) {
    if (auto ErrorOrBuffer = FileMgr.getBufferForFile(SourcePath))
      Result->SourceMgr.AddNewSourceBuffer(std::move(ErrorOrBuffer.get()),
                                           SMLoc());
  }

  Result->Module = Generator.getGeneratedModule();

  return Result;
}

LogicalResult tauCCMain(int Argc, const char **Argv) {
  tau::chx::registerUseOfUninitChecker();
  tau::core::CheckerCLParser CheckersOptions(CheckersCategory);
  cl::HideUnrelatedOptions({&TauCategory, &CheckersCategory});

  cl::SetVersionPrinter([](raw_ostream &OS) {
    // TODO: remove hardcoded version number
    OS << "tau C/C++ compiler v0.0.1\n";
  });

  auto IR = runClang(Argc, Argv);
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
