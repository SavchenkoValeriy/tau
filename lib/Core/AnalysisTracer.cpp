#include "tau/Core/AnalysisTracer.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace {
struct OpIndex {
  unsigned BlockIndex;
  unsigned InstIndex;
};
} // end anonymous namespace

namespace tau::core {

class Serializer::Implementation {
public:
  template <class T> llvm::json::Value serialize(const T &) const;

  llvm::DenseMap<mlir::Operation *, OpIndex> OpEnumerator;
};

#ifdef ANALYSIS_TRACER_ENABLED
namespace {
static llvm::cl::opt<std::string> TraceFunction("trace", llvm::cl::Hidden);
static llvm::cl::opt<std::string> TraceDir("trace-dir", llvm::cl::Hidden,
                                           llvm::cl::init(".tau"));
} // end anonymous namespace

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

  Serializer &getSerializer() { return SerializerInstance; }

  void addEvent(llvm::json::Value Event) { Trace.push_back(Event); }

  bool ShouldTrace;

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
      const unsigned BlockIndex = Indices[&Block];
      BlockObj["name"] = "bb" + std::to_string(BlockIndex);

      llvm::json::Array CodeArray;
      CodeArray.reserve(Block.getOperations().size());

      for (mlir::Operation &Op : Block.getOperations()) {
        std::string OpStr;
        llvm::raw_string_ostream OS(OpStr);
        Op.print(OS);
        CodeArray.push_back(OS.str());

        OpEnumerator[&Op] = {.BlockIndex = BlockIndex,
                             .InstIndex =
                                 static_cast<unsigned>(CodeArray.size() - 1)};
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

    SerializerInstance.PImpl->OpEnumerator = std::move(OpEnumerator);
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

    Root["trace"] = std::move(Trace);
    OutFile << llvm::json::Value(std::move(Root));
  }

  mlir::func::FuncOp Function;
  llvm::json::Object Root;
  llvm::json::Array Trace;
  llvm::DenseMap<mlir::Operation *, OpIndex> OpEnumerator;
  Serializer SerializerInstance;
};

AnalysisTracer::AnalysisTracer(mlir::func::FuncOp &Function)
    : PImpl(std::make_unique<Implementation>(Function)) {}

AnalysisTracer::~AnalysisTracer() = default;

Serializer &AnalysisTracer::getSerializer() { return PImpl->getSerializer(); }

void AnalysisTracer::addEvent(llvm::json::Value Event) {
  PImpl->addEvent(Event);
}

bool AnalysisTracer::shouldTrace() const { return PImpl->ShouldTrace; }
#endif

template <>
llvm::json::Value
Serializer::Implementation::serialize(mlir::Operation *const &Op) const {
  const auto It = OpEnumerator.find(Op);
  assert(It != OpEnumerator.end());
  const auto Position = It->getSecond();
  return llvm::json::Array({Position.BlockIndex, Position.InstIndex});
}

template <>
llvm::json::Value
Serializer::Implementation::serialize(const mlir::Value &Value) const {
  if (auto *Op = Value.getDefiningOp())
    return serialize(Op);

  mlir::BlockArgument Arg = llvm::cast<mlir::BlockArgument>(Value);
  std::string Buffer;
  llvm::raw_string_ostream SS(Buffer);
  SS << llvm::format("%%arg%d", Arg.getArgNumber());
  return SS.str();
}

Serializer::Serializer() : PImpl(std::make_unique<Implementation>()) {}

Serializer::~Serializer() = default;

template <class T>
llvm::json::Value Serializer::serialize(const T &Object) const {
  return PImpl->serialize(Object);
}

template <>
llvm::json::Value Serializer::serialize(mlir::Operation *const &Op) const {
  return PImpl->serialize(Op);
}
template <>
llvm::json::Value Serializer::serialize(const mlir::Value &Value) const {
  return PImpl->serialize(Value);
}

} // namespace tau::core
