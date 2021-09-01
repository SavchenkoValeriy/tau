#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"
#include "llvm/Support/CommandLine.h"

#include <clang/Tooling/CommonOptionsParser.h>

using namespace clang;
using namespace llvm;
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
  return Result;
}
