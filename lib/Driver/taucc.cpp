#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/Tooling/CommonOptionsParser.h>

using namespace clang;
using namespace llvm;

cl::OptionCategory TauCategory("tau compiler options");

int main(int Argc, const char **Argv) {
  tooling::CommonOptionsParser OptionsParser(Argc, Argv, TauCategory);
  tooling::ClangTool Tool(OptionsParser.getCompilations(),
                          OptionsParser.getSourcePathList());
  tau::frontend::AIRGenAction Generator;
  bool Result = Tool.run(&Generator);
  auto Module = Generator.getGeneratedModule();
  Module.dump();
  return Result;
}
