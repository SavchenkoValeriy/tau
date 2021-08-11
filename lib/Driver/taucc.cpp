#include <clang/Tooling/CommonOptionsParser.h>

using namespace clang;
using namespace llvm;

cl::OptionCategory TauCategory("tau compiler options");

int main(int Argc, const char **Argv) {
  tooling::CommonOptionsParser OptionsParser(Argc, Argv, TauCategory);
  return 0;
}
