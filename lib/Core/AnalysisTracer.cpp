#include "tau/Core/AnalysisTracer.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>

#include <fstream>

namespace {
static llvm::cl::opt<std::string> TraceFunction("trace", llvm::cl::Hidden);
static llvm::cl::opt<std::string> TraceDir("trace-dir", llvm::cl::Hidden,
                                           llvm::cl::init(".tau"));
} // end anonymous namespace

namespace tau::core {

class AnalysisTracer::Implementation {
public:
  Implementation(mlir::func::FuncOp &Function)
      : Function(Function), ShouldTrace(false) {
    if (!TraceFunction.empty() && Function.getName().contains(TraceFunction)) {
      ShouldTrace = true;
      serializeFunction();
    }
  }

  ~Implementation() {
    if (ShouldTrace)
      writeJSON();
  }

private:
  void serializeFunction() {
    llvm::json::Object FunctionObj;
    FunctionObj["name"] = Function.getName().str();
    llvm::json::Array BlocksArray;

    llvm::SmallDenseMap<mlir::Block *, unsigned> Indices;
    for (auto It : llvm::enumerate(Function.getBlocks())) {
      Indices[&It.value()] = It.index();
    }

    for (mlir::Block &Block : Function.getBlocks()) {
      llvm::json::Object BlockObj;
      BlockObj["name"] = "bb" + std::to_string(Indices[&Block]);

      llvm::json::Array CodeArray;
      for (mlir::Operation &Op : Block.getOperations()) {
        std::string OpStr;
        llvm::raw_string_ostream OS(OpStr);
        Op.print(OS);
        CodeArray.push_back(OS.str());
      }
      BlockObj["code"] = std::move(CodeArray);

      llvm::json::Array EdgesArray;
      for (mlir::Block *Succ : Block.getSuccessors()) {
        EdgesArray.push_back(Indices[Succ]);
      }
      BlockObj["edges"] = std::move(EdgesArray);

      BlocksArray.push_back(std::move(BlockObj));
    }

    FunctionObj["blocks"] = std::move(BlocksArray);
    Root["func"] = std::move(FunctionObj);
  }

  void writeJSON() {
    std::string Filename = Function.getName().str();
    std::replace(Filename.begin(), Filename.end(), ' ', '_');
    Filename.erase(std::remove(Filename.begin(), Filename.end(), '*'),
                   Filename.end());
    Filename.erase(std::remove(Filename.begin(), Filename.end(), '&'),
                   Filename.end());
    Filename += ".json";

    std::error_code EC = llvm::sys::fs::create_directory(TraceDir);
    if (EC) {
      llvm::errs() << "Error creating directory: " << EC.message() << "\n";
      return;
    }

    std::string FullPath = (llvm::Twine(TraceDir) + "/" + Filename).str();
    llvm::raw_fd_ostream OutFile(FullPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      return;
    }

    OutFile << llvm::json::Value(std::move(Root));
  }

  mlir::func::FuncOp Function;
  bool ShouldTrace;
  llvm::json::Object Root;
};

AnalysisTracer::AnalysisTracer(mlir::func::FuncOp &Function)
    : PImpl(std::make_unique<Implementation>(Function)) {}

AnalysisTracer::~AnalysisTracer() = default;

} // namespace tau::core
