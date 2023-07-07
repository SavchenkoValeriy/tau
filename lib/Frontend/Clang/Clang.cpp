#include "tau/Frontend/Clang/Clang.h"

#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/Error.h>

using namespace tau::frontend;
using namespace llvm;
using namespace clang;

namespace {} // end anonymous namespace

namespace tau::frontend {
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

std::unique_ptr<Output> runClang(int Argc, const char **Argv,
                                 const llvm::cl::list<std::string> &Sources,
                                 Options Opts) {
  auto CDB = readClangOptions(Argc, Argv);
  if (auto E = CDB.takeError()) {
    llvm::errs() << toString(std::move(E));
    return nullptr;
  }

  return runClang(*CDB->get(), Sources, Opts);
}

std::unique_ptr<Output> runClang(const clang::tooling::CompilationDatabase &CDB,
                                 ArrayRef<std::string> Sources, Options Opts) {
  auto Result = std::make_unique<Output>();

  tooling::ClangTool Tool(CDB, Sources);
  AIRGenAction Generator{Result->Context, Opts};
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

std::unique_ptr<Output> runClangOnCode(const llvm::Twine &Code,
                                       const std::vector<std::string> &Args,
                                       const llvm::Twine &FileName,
                                       Options Opts) {
  auto Result = std::make_unique<Output>();
  AIRGenAction Generator{Result->Context, Opts};

  tooling::runToolOnCodeWithArgs(Generator.create(), Code, Args, FileName,
                                 "tau-cc");

  Result->Module = Generator.getGeneratedModule();
  return Result;
}

std::unique_ptr<Output> runClangOnCode(const llvm::Twine &Code,
                                       const llvm::Twine &FileName,
                                       Options Opts) {
  auto Result = std::make_unique<Output>();
  AIRGenAction Generator{Result->Context, Opts};

  tooling::runToolOnCode(Generator.create(), Code, FileName);

  Result->Module = Generator.getGeneratedModule();
  return Result;
}

} // end namespace tau::frontend
