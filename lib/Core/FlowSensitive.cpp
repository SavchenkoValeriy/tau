#include "tau/Core/FlowSensitive.h"
#include "tau/AIR/AirAttrs.h"
#include "tau/AIR/StateID.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/FlowWorklist.h"
#include "tau/Core/State.h"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/AnalysisManager.h>

#include <immer/map.hpp>

#include <memory>
#include <utility>

using namespace tau;
using namespace air;
using namespace core;
using namespace mlir;
using namespace llvm;

namespace {

struct StateKey {
  StringRef CheckerID;
  StateID State;

  bool operator==(const StateKey &Other) const {
    return CheckerID == Other.CheckerID && State == Other.State;
  }
};

struct StateEvent {
  StateKey Key;
  Operation *Location = nullptr;
  const StateEvent *Parent = nullptr;
};
} // end anonymous namespace

namespace llvm {
inline hash_code
hash_value(StateID ID) { // NOLINT(readability-identifier-naming)
  return hash_value(StateID::Raw(ID));
}
} // end namespace llvm

namespace std {
template <> struct hash<StateKey> {
  size_t operator()(const StateKey &Key) const {
    return llvm::hash_combine(llvm::hash_value(Key.State), Key.CheckerID);
  }
};
template <> struct hash<Value> {
  size_t operator()(const Value &V) const {
    return llvm::hash_value(V.getAsOpaquePointer());
  }
};
} // end namespace std

namespace {
class Events {
private:
  using EventSet = immer::map<StateKey, const StateEvent *>;

  Events(EventSet Events) : AssociatedEvents(Events) {}

  EventSet AssociatedEvents;

public:
  Events() = default;
  Events(const StateEvent &Event) {
    AssociatedEvents =
        AssociatedEvents.insert(std::make_pair(Event.Key, &Event));
  }

  static Events join(const Events &LHS, const Events &RHS) {
    if (LHS.AssociatedEvents.size() < RHS.AssociatedEvents.size())
      return join(RHS, LHS);

    EventSet Result = LHS.AssociatedEvents;

    for (const auto &Event : RHS.AssociatedEvents)
      Result = Result.insert(Event);

    return Result;
  }

  Events join(const Events &Other) const { return join(*this, Other); }

  bool operator==(const Events &Other) const {
    return AssociatedEvents == Other.AssociatedEvents;
  }

  [[nodiscard]] bool contains(StringRef CheckerID, StateID ID) const {
    return AssociatedEvents.count({CheckerID, ID});
  }

  [[nodiscard]] const StateEvent *find(StringRef CheckerID, StateID ID) const {
    auto *const *Result = AssociatedEvents.find({CheckerID, ID});
    return Result ? *Result : nullptr;
  }

  Events replace(const StateEvent &Old, const StateEvent &New) const {
    EventSet Result = AssociatedEvents;
    Result = Result.erase(Old.Key);
    return Result.insert(std::make_pair(New.Key, &New));
  }

  bool hasNoCheckerState(StringRef CheckerID) const {
    return llvm::none_of(*this, [CheckerID](const auto &KeyEventPair) {
      return KeyEventPair.first.CheckerID == CheckerID;
    });
  }

  using const_iterator = EventSet::iterator;
  const_iterator begin() const { return AssociatedEvents.begin(); }
  const_iterator end() const { return AssociatedEvents.end(); }
};
} // end anonymous namespace

class FlowSensitiveAnalysis::Implementation {
  using ValueEvents = immer::map<Value, Events>;
  using BlockStateMap = SmallVector<ValueEvents, 20>;

  // State management
  BlockStateMap States;
  ValueEvents CurrentState;

  // Traversal instruments
  PostOrderBlockEnumerator &Enumerator;
  ForwardWorklist &Worklist;
  BitVector Processed;

  // Memory management
  llvm::SpecificBumpPtrAllocator<StateEvent> Allocator;

public:
  Implementation(FuncOp Function, AnalysisManager &AM)
      : Enumerator(AM.getAnalysis<PostOrderBlockEnumerator>()),
        Worklist(AM.getAnalysis<ForwardWorklist>()),
        Processed(Function.getBlocks().size()) {
    States.insert(States.end(), Function.getBlocks().size(), ValueEvents{});
    Worklist.enqueue(&Function.getBlocks().front());
  }

  void run() {
    while (Block *BB = Worklist.dequeue()) {
      visitBlock(*BB);
    }
  }

  void visitBlock(Block &BB) {
    CurrentState = joinPreds(BB);
    for (Operation &Op : BB) {
      visitOperation(Op);
    }
    if (setState(BB, CurrentState) == ChangeResult::Change ||
        !isProcessed(BB)) {
      markProcessed(BB);
      for (Block *Succ : BB.getSuccessors())
        Worklist.enqueue(Succ);
    }
  }

  ValueEvents joinPreds(Block &BB) const {
    ValueEvents Result;
    for (Block *Pred : BB.getPredecessors()) {
      Result = join(Result, getState(*Pred));
    }
    return Result;
  }

  static ValueEvents join(const ValueEvents &LHS, const ValueEvents &RHS) {
    if (LHS.size() < RHS.size())
      return join(RHS, LHS);

    ValueEvents Result = LHS;
    for (auto &[V, E] : RHS) {
      Result = Result.insert(std::make_pair(V, Events::join(Result[V], E)));
    }
    return Result;
  }

  void visitOperation(Operation &Op) {
    // Let's go over state attributes, the attributes marking what
    // should happen with all the values involved.
    auto StateAttrs = getStateAttributes(&Op);
    for (auto StateAttr : StateAttrs) {
      // State change:
      //    One of the values should change its state
      if (auto StateChange = StateAttr.dyn_cast<StateChangeAttr>()) {
        // Let's get current states of the value directly mentioned
        // in the attribute.
        Value AffectedValue = getOperandByIdx(Op, StateChange.getOperandIdx());
        StringRef CheckerID = StateChange.getCheckerID();
        Optional<StateID> From = StateChange.getFromState();
        StateID To = StateChange.getToState();

        if (const StateEvent *NewEvent =
                addTransition(AffectedValue, Op, CheckerID, From, To))
          // FIXME: this functionality doesn't belong here
          if (To.isError()) {
            auto &Checker = findChecker(CheckerID);
            InFlightDiagnostic Error = Checker.emitError(&Op, To);

            for (const StateEvent *CurrentEvent = NewEvent->Parent;
                 CurrentEvent; CurrentEvent = CurrentEvent->Parent)
              Checker.emitNote(Error, CurrentEvent->Location,
                               CurrentEvent->Key.State);
          }
      }
    }
  }

  const StateEvent *addTransition(Value ForValue, Operation &Location,
                                  StringRef CheckerID, Optional<StateID> From,
                                  StateID To) {
    Events Current = CurrentState[ForValue];

    if (!From) {
      // The lack of the state here means that it's the initial
      // transition, and we should check that among the tracked
      // states of the value there are no states of this checker.
      if (Current.hasNoCheckerState(CheckerID)) {
        const StateEvent &NewEvent =
            allocateStateEvent(CheckerID, To, &Location);
        associate(ForValue, Current.join(Events(NewEvent)));
        return &NewEvent;
      }

      return nullptr;
    }

    if (const StateEvent *FromEvent = Current.find(CheckerID, *From)) {
      const StateEvent &ToEvent =
          allocateStateEvent(CheckerID, To, &Location, FromEvent);
      Events NewSetOfEvents = Current.replace(*FromEvent, ToEvent);
      associate(ForValue, NewSetOfEvents);
      return &ToEvent;
    }

    return nullptr;
  }

  static Value getOperandByIdx(Operation &Op, unsigned Index) {
    // Index equal to the number of operands means result.
    if (Op.getNumOperands() == Index) {
      assert(Op.getNumResults() && "Operation has no result");
      return Op.getResult(0);
    }

    return Op.getOperand(Index);
  }

  void associate(Value V, Events E) {
    CurrentState = CurrentState.insert(std::make_pair(V, E));
  }

  ChangeResult setState(Block &BB, ValueEvents NewState) {
    ValueEvents &Current = getState(BB);

    if (Current == NewState)
      return ChangeResult::NoChange;

    Current = NewState;
    return ChangeResult::Change;
  }

  ValueEvents &getState(Block &BB) { return States[index(BB)]; }
  const ValueEvents &getState(Block &BB) const { return States[index(BB)]; }

  bool isProcessed(const Block &BB) const { return Processed.test(index(BB)); }
  void markProcessed(const Block &BB) { Processed.set(index(BB)); }

  unsigned index(const Block &BB) const {
    return Enumerator.getPostOrderIndex(&BB);
  }

  template <class... Args>
  const StateEvent &allocateStateEvent(Args &&...Rest) {
    return *(new (Allocator.Allocate()) StateEvent{Rest...});
  }
};

//===----------------------------------------------------------------------===//
//                                  Interface
//===----------------------------------------------------------------------===//

FlowSensitiveAnalysis::FlowSensitiveAnalysis(Operation *TopLevelOp,
                                             AnalysisManager &AM) {
  assert(isa<FuncOp>(TopLevelOp) && "Only functions are supported");
  PImpl = std::make_unique<Implementation>(cast<FuncOp>(TopLevelOp), AM);
  PImpl->run();
}

FlowSensitiveAnalysis::~FlowSensitiveAnalysis() = default;
