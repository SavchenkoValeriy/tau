#include "tau/Core/ReachingDefs.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"
#include "tau/Core/AliasAnalysis.h"
#include "tau/Core/FlowWorklist.h"
#include "tau/Core/PointsToAnalysis.h"
#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

#include <immer/map.hpp>
#include <immer/map_transient.hpp>

#include <utility>

using mlir::AnalysisManager;
using mlir::Operation;
using mlir::Value;

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace mlir::dataflow;
using namespace tau;
using namespace tau::core;

using Definitions = ReachingDefs::Definitions;
using CondensedDefs = BitVector;

namespace std {
template <> struct hash<Value> {
  size_t operator()(const Value &V) const {
    return llvm::hash_value(V.getAsOpaquePointer());
  }
};
} // end namespace std

namespace {

class State {
public:
  State() = default;

  void join(const State &RHS) {
    auto Transient = Map.transient();
    for (auto &KeyValuePair : RHS.Map) {
      Value V = KeyValuePair.first;
      const CondensedDefs &Defs = KeyValuePair.second;
      // It is insane, but I cannot capture Defs when I use
      // structured binding, so it is what it is.
      Transient.update(V, [&Defs](const CondensedDefs &Current) {
        CondensedDefs Result = Defs;
        if (Current.empty()) {
          return Result;
        }
        Result |= Current;
        return Result;
      });
    }
    Map = Transient.persistent();
  }

  [[nodiscard]] CondensedDefs getDefs(Value Of) const {
    if (const auto *It = Map.find(Of))
      return *It;
    return {};
  }

  [[nodiscard]] ChangeResult set(Value Address, unsigned DefIndex) {
    ChangeResult Result;
    Map = Map.update(Address, [DefIndex, &Result](const CondensedDefs &Defs) {
      CondensedDefs NewDefs;
      NewDefs.resize(DefIndex + 1);
      NewDefs.set(DefIndex);
      Result =
          (NewDefs == Defs) ? ChangeResult::NoChange : ChangeResult::Change;
      return NewDefs;
    });

    return Result;
  }

  [[nodiscard]] ChangeResult unset(Value Address) {
    ChangeResult Result;
    Map = Map.update(Address, [&Result](const CondensedDefs &Defs) {
      CondensedDefs NewDefs;
      Result =
          (NewDefs == Defs) ? ChangeResult::NoChange : ChangeResult::Change;
      return NewDefs;
    });
    return Result;
  }

  void print(raw_ostream &OS) const {
    OS << "{ ";
    llvm::interleaveComma(Map, OS, [&OS](auto KeyPair) {
      auto &[V, Defs] = KeyPair;
      OS << V << ": [";
      for (size_t Index = 0, End = Defs.size(); Index < End; ++Index) {
        OS << Defs.test(Index);
      }
      OS << "]";
    });
    OS << "}";
  }

  void dump() const { print(llvm::errs()); }

  [[nodiscard]] bool operator==(const State &RHS) const = default;

private:
  using ValueToDefs = immer::map<Value, CondensedDefs>;
  State(ValueToDefs Map) : Map(Map) {}

  ValueToDefs Map;
};

} // end anonymous namespace

class ReachingDefs::Implementation {
public:
  Implementation(FuncOp Function, AnalysisManager &AM);

  Definitions getDefinitions(Operation &At, mlir::Value For) const;

  void run() {
    initDefinitions(Function);
    Worklist.enqueue(&Function.getBlocks().front());
    while (Block *BB = Worklist.dequeue())
      if (visit(*BB) == ChangeResult::Change)
        Worklist.enqueueSuccessors(*BB);
  }

  [[nodiscard]] unsigned getIndex(air::StoreOp Store) const {
    return getIndex(Store.what());
  }
  [[nodiscard]] unsigned getIndex(Value Of) const {
    return DefToIndex.lookup(Of);
  }

  [[nodiscard]] Value lookupDefinition(unsigned Index) const {
    assert(Index < AllDefinitions.size());
    return AllDefinitions[Index];
  }

private:
  void initDefinitions(FuncOp Function);

  [[nodiscard]] inline ChangeResult visit(Operation &Op);
  [[nodiscard]] inline ChangeResult visit(Block &BB);

  [[nodiscard]] inline State joinPreds(Block &BB);

  [[nodiscard]] inline ChangeResult handleEscapes(CallOp Call);

  [[nodiscard]] State &lookupState(Block &BB) { return States[&BB]; }
  [[nodiscard]] State &lookupState(Operation &Op) { return States[&Op]; }

  [[nodiscard]] State lookupState(Operation &Op) const {
    if (const auto It = States.find(&Op); It != States.end())
      return It->getSecond();
    return {};
  }
  [[nodiscard]] ChangeResult updateState(Operation &Op) {
    return updateState(&Op);
  }
  [[nodiscard]] ChangeResult updateState(Block &BB) { return updateState(&BB); }
  [[nodiscard]] inline ChangeResult updateState(ProgramPoint P);

  [[nodiscard]] static Value getAddress(air::StoreOp Store) {
    Value Address = Store.where();
    if (auto Ref = dyn_cast<air::RefOp>(Address.getDefiningOp()))
      return Ref.value();
    return Address;
  }

  llvm::DenseMap<Value, unsigned> DefToIndex;
  std::vector<Value> AllDefinitions;

  State CurrentState;
  llvm::DenseMap<ProgramPoint, State> States;

  FuncOp Function;

  ForwardWorklist &Worklist;
  AliasAnalysis &AA;
  PointsToAnalysis &PA;
};

ReachingDefs::Implementation::Implementation(FuncOp Function,
                                             AnalysisManager &AM)
    : Function(Function), Worklist(AM.getAnalysis<ForwardWorklist>()),
      AA(AM.getAnalysis<AliasAnalysis>()),
      PA(AM.getAnalysis<PointsToAnalysis>()) {}

void ReachingDefs::Implementation::initDefinitions(FuncOp Function) {
  Function.walk([this](air::StoreOp Store) {
    DefToIndex[Store.what()] = AllDefinitions.size();
    AllDefinitions.push_back(Store.what());
  });
}

ChangeResult ReachingDefs::Implementation::visit(Operation &Op) {
  ChangeResult Result = updateState(Op);

  llvm::TypeSwitch<Operation *, void>(&Op)
      .Case([this, &Result](air::StoreOp Store) {
        Value Modified = getAddress(Store);
        Result |= CurrentState.set(Modified, getIndex(Store));

        for (auto AlsoModified : AA.getAliases(Modified))
          Result |= CurrentState.unset(AlsoModified);
      })
      .Case([this, &Result](CallOp Call) { Result |= handleEscapes(Call); });

  return Result;
}

ChangeResult ReachingDefs::Implementation::visit(Block &BB) {
  CurrentState = joinPreds(BB);
  for (Operation &Op : BB)
    if (visit(Op) == ChangeResult::NoChange)
      return ChangeResult::NoChange;
  return updateState(BB);
}

State ReachingDefs::Implementation::joinPreds(Block &BB) {
  State Result;
  for (Block *Pred : BB.getPredecessors())
    Result.join(lookupState(*Pred));

  return Result;
}

ChangeResult ReachingDefs::Implementation::handleEscapes(CallOp Call) {
  ChangeResult Result;
  std::queue<Value> Escapes;
  for (Value Arg : Call.getArgOperands()) {
    if (Arg.getType().isa<air::PointerType>()) {
      Escapes.push(Arg);
      for (Value ArgAlias : AA.getAliases(Arg))
        Result |= CurrentState.unset(ArgAlias);
    }
  }

  while (!Escapes.empty()) {
    Value EscapedValue = Escapes.front();
    Escapes.pop();

    Result |= CurrentState.unset(EscapedValue);

    Type PointeeType =
        EscapedValue.getType().cast<air::PointerType>().getElementType();

    if (!PointeeType.isa<air::PointerType>())
      continue;

    for (Value Pointee : PA.getPointsToSet(EscapedValue))
      Escapes.push(Pointee);
  }
  return Result;
}

Definitions
ReachingDefs::Implementation::getDefinitions(Operation &At,
                                             mlir::Value For) const {
  Definitions Result;
  const State &Element = lookupState(At);
  const CondensedDefs CDs = Element.getDefs(For);

  for (size_t I = 0, E = CDs.size(); I < E; ++I)
    if (CDs.test(I))
      Result.push_back(lookupDefinition(I));

  return Result;
}

ChangeResult ReachingDefs::Implementation::updateState(ProgramPoint P) {
  auto [It, Inserted] = States.try_emplace(P, CurrentState);
  // Haven't visited this point before - changed.
  if (Inserted)
    return ChangeResult::Change;
  if (It->getSecond() != CurrentState) {
    It->getSecond() = CurrentState;
    return ChangeResult::Change;
  }
  return ChangeResult::NoChange;
}

//===----------------------------------------------------------------------===//
//                                  Interface
//===----------------------------------------------------------------------===//

ReachingDefs::ReachingDefs(Operation *FunctionOp, AnalysisManager &AM) {
  assert(isa<FuncOp>(FunctionOp) && "Only functions are supported");
  FuncOp Function = cast<FuncOp>(FunctionOp);
  PImpl = std::make_unique<Implementation>(Function, AM);
  if (!Function.isDeclaration())
    PImpl->run();
}

ReachingDefs::Definitions ReachingDefs::getDefinitions(Operation &At,
                                                       mlir::Value For) const {
  return PImpl->getDefinitions(At, For);
}

ReachingDefs::ReachingDefs(ReachingDefs &&) = default;
ReachingDefs &ReachingDefs::operator=(ReachingDefs &&) = default;

ReachingDefs::~ReachingDefs() = default;
