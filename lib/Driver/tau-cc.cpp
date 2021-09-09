#include "tau/Checkers/Checkers.h"
#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/Tooling/CompilationDatabase.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>

using namespace clang;
using namespace llvm;
using namespace mlir;

namespace {
cl::OptionCategory TauCategory("tau compiler options");
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

LogicalResult tauCCMain(int Argc, const char **Argv) {
  tau::chx::registerUseOfUninitChecker();

  // TODO: reimplement it to surface only relevant options
  PassPipelineCLParser PassPipeline("", "Checkers to run");
  cl::SetVersionPrinter([](raw_ostream &OS) {
    // TODO: remove hardcoded version number
    OS << "tau C/C++ compiler v0.0.1\n";
  });

  auto CDB = readClangOptions(Argc, Argv);
  if (auto E = CDB.takeError()) {
    llvm::errs() << toString(std::move(E));
    return failure();
  }
  tooling::ClangTool Tool(*CDB->get(), SourcePaths);
  tau::frontend::AIRGenAction Generator;
  if (Tool.run(&Generator))
    // TODO: output clang errors?
    return failure();

  auto Module = Generator.getGeneratedModule();

  if (DumpAction == DumpTarget::AIR)
    Module.dump();

  MLIRContext &Context = Generator.getContext();

  PassManager PM(&Context);
  OpPassManager &FPM = PM.nest<FuncOp>();

  auto ErrorHandler = [&](const Twine &Message) { return failure(); };

  return success(succeeded(PassPipeline.addToPipeline(FPM, ErrorHandler)) &&
                 succeeded(PM.run(Module)));
}

int main(int Argc, const char **Argv) { return failed(tauCCMain(Argc, Argv)); }
