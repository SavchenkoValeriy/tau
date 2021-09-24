#include "tau/Core/FlowSensitive.h"

#include "tau/AIR/AirAttrs.h"
#include "tau/AIR/StateID.h"
#include "tau/Core/FlowWorklist.h"
#include "tau/Core/State.h"
#include "tau/Core/StateEventForest.h"

#include <llvm/ADT/ArrayRef.h>
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

  [[nodiscard]] static Events join(const Events &LHS, const Events &RHS) {
    // It's easier to merge smaller sets into larger ones.
    if (LHS.AssociatedEvents.size() < RHS.AssociatedEvents.size())
      return join(RHS, LHS);

    EventSet Result = LHS.AssociatedEvents;

    // If Result already has an event with the same key - doesn't matter.
    // We can pick just one to carry on further.
    //
    // NOTE: This in fact might cause a problem if the event that we discard
    //       is possible and the one that we kept is not.
    for (const auto &Event : RHS.AssociatedEvents)
      Result = Result.insert(Event);

    return Result;
  }

  [[nodiscard]] Events join(const Events &Other) const {
    return join(*this, Other);
  }

  [[nodiscard]] bool operator==(const Events &Other) const {
    return AssociatedEvents == Other.AssociatedEvents;
  }

  [[nodiscard]] bool contains(StringRef CheckerID, StateID ID) const {
    return AssociatedEvents.count({CheckerID, ID});
  }

  [[nodiscard]] const StateEvent *find(StringRef CheckerID, StateID ID) const {
    const auto *const *Result = AssociatedEvents.find({CheckerID, ID});
    return Result ? *Result : nullptr;
  }

  [[nodiscard]] Events replace(const StateEvent &Old,
                               const StateEvent &New) const {
    EventSet Result = AssociatedEvents;
    Result = Result.erase(Old.Key);
    return Result.insert(std::make_pair(New.Key, &New));
  }

  [[nodiscard]] bool hasNoCheckerState(StringRef CheckerID) const {
    return llvm::none_of(*this, [CheckerID](const auto &KeyEventPair) {
      return KeyEventPair.first.CheckerID == CheckerID;
    });
  }

  using const_iterator = EventSet::iterator;
  const_iterator begin() const { return AssociatedEvents.begin(); }
  const_iterator end() const { return AssociatedEvents.end(); }
};
} // end anonymous namespace

namespace std {
template <> struct hash<Value> {
  size_t operator()(const Value &V) const {
    return llvm::hash_value(V.getAsOpaquePointer());
  }
};
} // end namespace std

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
  StateEventForest Forest;

  // Issue tracking
  SmallVector<Issue, 20> FoundIssues;

public:
  Implementation(FuncOp Function, AnalysisManager &AM)
      : Enumerator(AM.getAnalysis<PostOrderBlockEnumerator>()),
        Worklist(AM.getAnalysis<ForwardWorklist>()),
        Processed(Function.getBlocks().size()) {
    States.insert(States.end(), Function.getBlocks().size(), ValueEvents{});
    Worklist.enqueue(&Function.getBlocks().front());
  }

  void run() {
    while (Block *BB = Worklist.dequeue())
      visit(*BB);
  }

  StateEventForest &getForest() { return Forest; }
  ArrayRef<Issue> getIssues() { return FoundIssues; }

private:
  void visit(Block &BB) {
    // CurrentState contains the state of all values while we walk
    // through the basic block.  We start it with the disjunction
    // of all states from the block's predecessors.
    CurrentState = joinPreds(BB);

    // Sequentially visit all block's operation.
    // This visitation affects the CurrentState.
    for (Operation &Op : BB)
      visit(Op);

    // We have two options when we need to keep going and traverse
    // block's successors:
    //
    //    * The state at the block's exit has changed,
    //      meaning that it might change states of the successor blocks.
    //
    //    * It's the first time we visit this block, and it's
    //      successors are still to be processed at least once.
    if (setState(BB, CurrentState) == ChangeResult::Change ||
        !isProcessed(BB)) {
      markProcessed(BB);
      Worklist.enqueueSuccessors(BB);
    }
  }

  ValueEvents joinPreds(Block &BB) const {
    ValueEvents Result;
    for (Block *Pred : BB.getPredecessors())
      Result = join(Result, getState(*Pred));

    return Result;
  }

  static ValueEvents join(const ValueEvents &LHS, const ValueEvents &RHS) {
    // It's easier to merge smaller maps into larger ones.
    if (LHS.size() < RHS.size())
      return join(RHS, LHS);

    ValueEvents Result = LHS;
    for (auto &[V, E] : RHS)
      // Merge sets of events for the same values.
      // Here we have three possible situations:
      //
      //    * V is both in LHS and RHS
      //    This means that Result[V] == LHS[V], E == RHS[V], and we associate
      //    join(LHS[V], RHS[V]) with V.
      //
      //    * V is in RHS only
      //    In this situation, Result[V] is empty and after the next operation
      //    it becomes RHS[V].
      //
      //    * V is in LHS only
      //    In this situaution we won't iterate over this value at all,
      //    but Result[V] == LHS[V] prior to the loop, which is the correct
      //    answer.
      Result = Result.insert(std::make_pair(V, Events::join(Result[V], E)));

    return Result;
  }

  void visit(Operation &Op) {
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
                addTransition(AffectedValue, Op, CheckerID, From, To);
            NewEvent != nullptr && To.isError())
          // TODO: use domination relationship to figure out whether
          //       we can guarantee that the issue can happen.
          FoundIssues.push_back({*NewEvent, false});
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
        const StateEvent &NewEvent = Forest.addEvent(CheckerID, To, &Location);
        // Since the value had no checker-related events prior to this,
        // we can simply add a new event.
        associate(ForValue, Current.join(Events(NewEvent)));
        return &NewEvent;
      }

      return nullptr;
    }

    if (const StateEvent *FromEvent = Current.find(CheckerID, *From)) {
      const StateEvent &ToEvent =
          Forest.addEvent(CheckerID, To, &Location, FromEvent);
      // Since this is a state transition, we need to replace the previous
      // event.
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
    // Associating happens for the current basic block only.
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

StateEventForest &FlowSensitiveAnalysis::getEventForest() {
  return PImpl->getForest();
}

ArrayRef<FlowSensitiveAnalysis::Issue> FlowSensitiveAnalysis::getFoundIssues() {
  return PImpl->getIssues();
}
