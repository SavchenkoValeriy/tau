#include "tau/Checkers/Checkers.h"
#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/Tooling/CommonOptionsParser.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LogicalResult.h>

using namespace clang;
using namespace llvm;
using namespace mlir;

namespace {
cl::OptionCategory TauCategory("tau compiler options");
enum class DumpTarget { None, AST, AIR };
static cl::opt<DumpTarget>
    DumpAction("dump", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpTarget::AST, "ast", "dump the AST")),
               cl::values(clEnumValN(DumpTarget::AIR, "air", "dump the AIR")),
               cl::cat(TauCategory));

} // end anonymous namespace

int main(int Argc, const char **Argv) {
  tau::chx::registerUseOfUninitChecker();

  // TODO: Surface options from PassPipeline to --help
  PassPipelineCLParser PassPipeline("", "Checkers to run");

  auto OptionsParser =
      tooling::CommonOptionsParser::create(Argc, Argv, TauCategory);
  if (!OptionsParser)
    return 1;
  tooling::ClangTool Tool(OptionsParser->getCompilations(),
                          OptionsParser->getSourcePathList());
  tau::frontend::AIRGenAction Generator;
  bool Result = Tool.run(&Generator);
  auto Module = Generator.getGeneratedModule();

  if (DumpAction == DumpTarget::AIR)
    Module.dump();

  MLIRContext &Context = Generator.getContext();

  PassManager PM(&Context);
  OpPassManager &FPM = PM.nest<FuncOp>();

  auto ErrorHandler = [&](const Twine &Message) {
    emitError(UnknownLoc::get(&Context)) << Message;
    return failure();
  };

  if (failed(PassPipeline.addToPipeline(FPM, ErrorHandler)))
    return 1;

  if (failed(PM.run(Module)))
    return 1;

  return Result;
}
