#include "tau/Core/FlowSensitive.h"

#include "tau/AIR/AirAttrs.h"
#include "tau/AIR/StateID.h"
#include "tau/Core/AnalysisTracer.h"
#include "tau/Core/Events.h"
#include "tau/Core/FlowWorklist.h"
#include "tau/Core/MemoryStore.h"
#include "tau/Core/MutualExclusionAnalysis.h"
#include "tau/Core/State.h"
#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/JSON.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/AnalysisManager.h>

#include <immer/map.hpp>

#include <memory>
#include <utility>

using namespace tau;
using namespace air;
using namespace core;
using namespace mlir;
using namespace mlir::func;
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
        AssociatedEvents.insert(std::make_pair(Event.getKey(), &Event));
  }

  [[nodiscard]] static Events join(const Events &LHS, const Events &RHS) {
    if (LHS.AssociatedEvents.size() < RHS.AssociatedEvents.size())
      return joinImpl(RHS, LHS);

    return joinImpl(LHS, RHS);
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
    Result = Result.erase(Old.getKey());
    return Result.insert(std::make_pair(New.getKey(), &New));
  }

  [[nodiscard]] bool hasNoCheckerState(StringRef CheckerID) const {
    return llvm::none_of(*this, [CheckerID](const auto &KeyEventPair) {
      return KeyEventPair.first.CheckerID == CheckerID;
    });
  }

  using const_iterator = EventSet::iterator;
  const_iterator begin() const { return AssociatedEvents.begin(); }
  const_iterator end() const { return AssociatedEvents.end(); }

private:
  [[nodiscard]] static Events joinImpl(const Events &Larger,
                                       const Events &Smaller) {
    // It's easier to merge smaller sets into larger ones.
    EventSet Result = Larger.AssociatedEvents;

    // If Result already has an event with the same key - doesn't matter.
    // We can pick just one to carry on further.
    //
    // NOTE: This in fact might cause a problem if the event that we discard
    //       is possible and the one that we kept is not.
    for (const auto &Event : Smaller.AssociatedEvents)
      Result = Result.insert(Event);

    return Result;
  }
};
} // end anonymous namespace

namespace std {
template <> struct hash<Value> {
  size_t operator()(const Value &V) const {
    return llvm::hash_value(V.getAsOpaquePointer());
  }
};
} // end namespace std

namespace tau::core {
template <> llvm::json::Value Serializer::serialize(const Events &E) const {
  llvm::json::Array Result;
  for (auto &It : E) {
    llvm::json::Object Key{};
    Key["checker"] = It.first.CheckerID;
    Key["state"] = It.first.State.ID;
    Result.push_back(std::move(Key));
  }
  return Result;
}
} // end namespace tau::core

/// Implementation of the flow-sensitive analysis algorithm.
///
/// This class tracks the state of values across a function's control flow
/// graph, identifying potential issues based on state transitions defined by
/// checkers. It uses a worklist algorithm to propagate states through the CFG,
/// handling joins at confluence points and tracking the history of state
/// changes to determine if detected issues are guaranteed to occur.
///
/// For more reasoning behind the `isGuaranteed`, please refer to the
/// corresponding method and its comments.
class FlowSensitiveAnalysis::Implementation {
public:
  using ValueEvents = immer::map<Value, Events>;
  using BlockStateMap = SmallVector<ValueEvents, 20>;
  using BlockStoreMap = SmallVector<MemoryStore, 20>;

private:
  // Memory management
  EventHierarchy Hierarchy;

  // State management
  BlockStateMap States;
  BlockStoreMap Stores;
  ValueEvents CurrentState;
  MemoryStore CurrentStore;

  // Traversal instruments
  TopoOrderBlockEnumerator &Enumerator;
  ForwardWorklist &Worklist;
  BitVector Processed;

  // Issue tracking
  SmallVector<Issue, 20> FoundIssues;
  MutualExclusionAnalysis MutualExclusion;
  DominanceInfo &DomTree;
  PostDominanceInfo &PostDomTree;

  // Debugging
  AnalysisTracer Tracer;

public:
  Implementation(FuncOp Function, AnalysisManager &AM)
      : CurrentStore(Hierarchy),
        Enumerator(AM.getAnalysis<TopoOrderBlockEnumerator>()),
        Worklist(AM.getAnalysis<ForwardWorklist>()),
        Processed(Function.getBlocks().size()),
        MutualExclusion(AM.getAnalysis<MutualExclusionAnalysis>()),
        DomTree(AM.getAnalysis<DominanceInfo>()),
        PostDomTree(AM.getAnalysis<PostDominanceInfo>()), Tracer(Function) {
    States.insert(States.end(), Function.getBlocks().size(), ValueEvents{});
    Stores.insert(Stores.end(), Function.getBlocks().size(),
                  MemoryStore{Hierarchy});
    Worklist.enqueue(&Function.getBlocks().front());
  }

  void run() {
    while (Block *BB = Worklist.dequeue())
      visit(*BB);
  }

  EventHierarchy &getEventHierarchy() { return Hierarchy; }
  ArrayRef<Issue> getIssues() { return FoundIssues; }

private:
  void visit(Block &BB) {
    // CurrentState contains the state of all values while we walk
    // through the basic block.  We start it with the disjunction
    // of all states from the block's predecessors.
    std::tie(CurrentState, CurrentStore) = joinPreds(BB);

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
    const ChangeResult StateChanged = setState(BB, CurrentState);
    const ChangeResult StoreChanged = setStore(BB, CurrentStore);
    if (StateChanged == ChangeResult::Change ||
        StoreChanged == ChangeResult::Change || !isProcessed(BB)) {
      markProcessed(BB);
      Worklist.enqueueSuccessors(BB);
    }
  }

  std::pair<ValueEvents, MemoryStore> joinPreds(Block &BB) {
    ValueEvents Events;
    MemoryStore Store{Hierarchy};
    for (Block *Pred : BB.getPredecessors()) {
      Events = join(Events, getState(*Pred));
      Store = Store.join(getStore(*Pred));
    }

    return {Events, Store};
  }

  static ValueEvents join(const ValueEvents &LHS, const ValueEvents &RHS) {
    if (LHS.size() < RHS.size())
      return joinImpl(RHS, LHS);

    return joinImpl(LHS, RHS);
  }

  static ValueEvents joinImpl(const ValueEvents &Larger,
                              const ValueEvents &Smaller) {
    // It's easier to merge smaller maps into larger ones.
    ValueEvents Result = Larger;
    for (auto &[V, E] : Smaller)
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
    Tracer.recordBeforeState(&Op, CurrentState, CurrentStore);
    const auto RecordAfter = llvm::make_scope_exit([this, &Op]() {
      Tracer.recordAfterState(&Op, CurrentState, CurrentStore);
    });

    CurrentStore = CurrentStore.interpret(&Op);

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
        std::optional<StateID> From = StateChange.getFromState();
        StateID To = StateChange.getToState();

        for (MemoryStore::Definition CanonicalDefinition :
             CurrentStore.getDefininingValues(AffectedValue)) {
          if (const StateEvent *NewEvent =
                  addTransition(CanonicalDefinition, Op, CheckerID, From, To);
              NewEvent != nullptr && To.isError())
            addIssue(*NewEvent);
        }
      }
    }
  }

  const StateEvent *addTransition(MemoryStore::Definition ForValue,
                                  Operation &Location, StringRef CheckerID,
                                  std::optional<StateID> From, StateID To) {
    Events Current = CurrentState[ForValue.Value];

    // TODO: This is a very crude deduplication method, but it helps to
    //       prevent infinite loops with re-creating the same events over
    //       and over.
    if (Current.contains(CheckerID, To))
      return nullptr;

    if (!From) {
      // The lack of the state here means that it's the initial
      // transition, and we should check that among the tracked
      // states of the value there are no states of this checker.
      if (Current.hasNoCheckerState(CheckerID)) {
        const StateEvent *NewEvent = nullptr;
        if (ForValue.Event) {
          NewEvent = &Hierarchy.addStateEvent({CheckerID, To}, &Location,
                                              {ForValue.Event});
        } else {
          NewEvent = &Hierarchy.addStateEvent({CheckerID, To}, &Location);
        }

        // Since the value had no checker-related events prior to this,
        // we can simply add a new event.
        associate(ForValue.Value, Current.join(Events(*NewEvent)));
        return NewEvent;
      }

      return nullptr;
    }

    if (const StateEvent *FromEvent = Current.find(CheckerID, *From)) {
      const StateEvent *ToEvent = nullptr;
      if (ForValue.Event) {
        if (EventHierarchy::areMutuallyExclusive(FromEvent, ForValue.Event,
                                                 MutualExclusion, Enumerator))
          return nullptr;

        ToEvent = &Hierarchy.addStateEvent({CheckerID, To}, &Location,
                                           {FromEvent, ForValue.Event});
      } else {
        ToEvent =
            &Hierarchy.addStateEvent({CheckerID, To}, &Location, {FromEvent});
      }

      // Since this is a state transition, we need to replace the previous
      // event.
      Events NewSetOfEvents = Current.replace(*FromEvent, *ToEvent);
      associate(ForValue.Value, NewSetOfEvents);
      return ToEvent;
    }

    return nullptr;
  }

  void addIssue(const StateEvent &ErrorEvent) {
    const auto LinearizedEvents =
        Hierarchy.linearizeChainOfEvents(&ErrorEvent, Enumerator);
    const bool Guaranteed = isGuaranteed(ErrorEvent, LinearizedEvents);

    FoundIssues.push_back({std::move(LinearizedEvents), Guaranteed});
  }

  bool isGuaranteed(const StateEvent &ErrorEvent,
                    const LinearChainOfEvents &LinearizedEvents) const {
    size_t CurrentIndex = 1;

    AbstractEvent Prev = &ErrorEvent, Current = Prev;

    // This part of the algorithm checks whether the flow-sensitive
    // framework is enough to report the issue.  It uses (post-)domination
    // relationship to figure this out.
    //
    // The key assumption that we make here is that every condition
    // and every branch in the code that are written, are actually possible.
    // If the error "occurs" in the dead part of the code, and the user
    // relies on it being dead, there is no reason in keeping that code.
    //
    // The main problem in here is to detect the situations when events
    // appear on mutually exclusive paths, ie we need to make contradictory
    // assumptions.  In order to prevent this, we forbid consecutive non-nested
    // assumptions.
    //
    // But first let's start with (post-)domination.  If event A dominates B,
    // and B happened, A also happened.  If event B post-dominates A, and A
    // happened, B also happened.  Thus, if all events in the chain dominate and
    // post-dominate one another, they are guaranteed to happen together and we
    // can always assume that the topmost event can happen.
    //
    // This condition can be significantly relaxed though.  If we assume that
    // the topmost event happens, we can simply check for post-domination of
    // all the following events, since it will also guarantee us that they
    // will follow no matter what concrete execution we are in.
    //
    // In order to understand the next relaxation of the rule, let's consider
    // the following example:
    //
    // ```
    // if (x) {
    //   ... // event A
    //   if (y) {
    //     ... // event B
    //     if (z) {
    //       ... // event C
    //     }
    //   }
    // }
    // ```
    //
    // What can we say about code like this?  Can we safely assume that
    // conditions `x`, `y`, and `z` are true at the same time?  Well, yes!
    // We can do this because if they couldn't be true at the same time, this
    // code wouldn't've been structured like this.  We can also notice that
    // events in such conditions dominate one another (A dominates B,
    // B dominates C).
    //
    // Joining this reasoning with out previous solution, we can say that
    // in the chain of events { A_1, A_2, ... , A_n } if there exists
    // 1 <= k <= n such that A_i dominates A_i+1 for all i < k, and
    // A_i post-dominates A_i+1 for all k <= i < n, then we can guarantee
    // that the given chain doesn't include mutually exclusive assumptions.
    for (; CurrentIndex < LinearizedEvents.size();
         Prev = Current, ++CurrentIndex) {
      // We start from the very last event of the chain, so in order to find
      // A_k, we need to check for post-dominance.
      Current = LinearizedEvents[CurrentIndex];
      if (!PostDomTree.postDominates(EventHierarchy::getLocationOf(Prev),
                                     EventHierarchy::getLocationOf(Current)))
        break;
    }

    for (; CurrentIndex < LinearizedEvents.size();
         Prev = Current, ++CurrentIndex) {
      // If we got here, we found an event A_i, so that A_i+1 doesn't
      // post-dominate it.  Now we need to "flip the switch" and check
      // for dominance.
      Current = LinearizedEvents[CurrentIndex];
      if (!DomTree.dominates(EventHierarchy::getLocationOf(Current),
                             EventHierarchy::getLocationOf(Prev)))
        return false;
    }

    return true;
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

  ChangeResult setStore(Block &BB, MemoryStore NewStore) {
    MemoryStore &Current = getStore(BB);

    if (Current == NewStore)
      return ChangeResult::NoChange;

    Current = NewStore;
    return ChangeResult::Change;
  }

  MemoryStore &getStore(Block &BB) { return Stores[index(BB)]; }
  const MemoryStore &getStore(Block &BB) const { return Stores[index(BB)]; }

  bool isProcessed(const Block &BB) const { return Processed.test(index(BB)); }
  void markProcessed(const Block &BB) { Processed.set(index(BB)); }

  unsigned index(const Block &BB) const {
    return Enumerator.getTopoOrderIndex(&BB);
  }
};

namespace tau::core {
template <>
llvm::json::Value Serializer::serialize(
    const FlowSensitiveAnalysis::Implementation::ValueEvents &E) const {
  llvm::json::Array Result;
  for (auto &It : E) {
    llvm::json::Object ValueState{};
    ValueState["value"] = serialize(It.first);
    ValueState["state"] = serialize(It.second);
    Result.push_back(std::move(ValueState));
  }
  return Result;
}
} // end namespace tau::core

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

EventHierarchy &FlowSensitiveAnalysis::getEventHierarchy() {
  return PImpl->getEventHierarchy();
}

ArrayRef<FlowSensitiveAnalysis::Issue> FlowSensitiveAnalysis::getFoundIssues() {
  return PImpl->getIssues();
}
