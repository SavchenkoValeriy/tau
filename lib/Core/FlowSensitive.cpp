#include "tau/Core/FlowSensitive.h"
#include "tau/AIR/AirAttrs.h"
#include "tau/AIR/StateID.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/State.h"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <immer/map.hpp>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
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
} // end namespace std

namespace {
class StateLatticeValue {
private:
  using StateSet = immer::map<StateKey, const StateEvent *>;

  StateLatticeValue(StateSet States) : AssociatedStates(States) {}

  StateSet AssociatedStates;

public:
  StateLatticeValue() = default;
  StateLatticeValue(const StateEvent &Event) {
    AssociatedStates =
        AssociatedStates.insert(std::make_pair(Event.Key, &Event));
  }

  static StateLatticeValue getPessimisticValueState(MLIRContext *) {
    return {};
  }

  static StateLatticeValue getPessimisticValueState(Value) { return {}; }

  static StateLatticeValue join(const StateLatticeValue &LHS,
                                const StateLatticeValue &RHS) {
    StateSet Result = LHS.AssociatedStates;

    for (const auto &Event : RHS.AssociatedStates)
      Result = Result.insert(Event);

    return Result;
  }

  bool operator==(const StateLatticeValue &Other) const {
    return AssociatedStates == Other.AssociatedStates;
  }

  [[nodiscard]] bool contains(StringRef CheckerID, StateID ID) const {
    return AssociatedStates.count({CheckerID, ID});
  }

  [[nodiscard]] const StateEvent *find(StringRef CheckerID, StateID ID) const {
    auto *const *Result = AssociatedStates.find({CheckerID, ID});
    return Result ? *Result : nullptr;
  }

  StateLatticeValue replace(const StateEvent &Old,
                            const StateEvent &New) const {
    StateSet Result = AssociatedStates;
    Result = Result.erase(Old.Key);
    return Result.insert(std::make_pair(New.Key, &New));
  }

  bool hasNoCheckerState(StringRef CheckerID) const {
    return llvm::none_of(*this, [CheckerID](const auto &KeyEventPair) {
      return KeyEventPair.first.CheckerID == CheckerID;
    });
  }

  using const_iterator = StateSet::iterator;
  const_iterator begin() const { return AssociatedStates.begin(); }
  const_iterator end() const { return AssociatedStates.end(); }
};
} // end anonymous namespace

class FlowSensitiveAnalysis::Implementation
    : public ForwardDataFlowAnalysis<StateLatticeValue> {
  using Base = ForwardDataFlowAnalysis<StateLatticeValue>;
  using AssociatedStates = LatticeElement<StateLatticeValue>;

  llvm::SpecificBumpPtrAllocator<StateEvent> Allocator;

public:
  using Base::ForwardDataFlowAnalysis;

  template <class... Args>
  const StateEvent &allocateStateEvent(Args &&...Rest) {
    return *(new (Allocator.Allocate()) StateEvent{Rest...});
  }

  ChangeResult visitOperation(Operation *Op,
                              ArrayRef<AssociatedStates *> OperandStates) {
    ChangeResult Result = ChangeResult::NoChange;

    // Let's go over state attributes, the attributes marking what
    // should happen with all the values involved.
    auto StateAttrs = getStateAttributes(Op);
    for (auto StateAttr : StateAttrs) {
      // State change:
      //    One of the values should change its state
      if (auto StateChange = StateAttr.dyn_cast<StateChangeAttr>()) {
        // Let's get current states of the value directly mentioned
        // in the attribute.
        auto &StatesOfTheChangingOperand =
            getOperandByIdx(Op, StateChange.getOperandIdx(), OperandStates);

        Result |= addTransition(
            StatesOfTheChangingOperand, Op, StateChange.getCheckerID(),
            StateChange.getFromState(), StateChange.getToState());
      }
    }

    if (Op->getNumResults()) {
      Value ResultValue = Op->getResult(0);
      AssociatedStates &ResultStates = getLatticeElement(ResultValue);
      if (ResultStates.isUninitialized())
        Result |= ResultStates.join(StateLatticeValue{});
    }

    return Result;
  }

  ChangeResult addTransition(AssociatedStates &CurrentStates,
                             Operation *Location, StringRef CheckerID,
                             Optional<StateID> From, StateID To) {
    ChangeResult Result = ChangeResult::NoChange;

    if (!From) {
      // The lack of the state here means that it's the initial
      // transition, and we should check that among the tracked
      // states of the value there are no states of this checker.
      if (CurrentStates.isUninitialized() ||
          CurrentStates.getValue().hasNoCheckerState(CheckerID))
        return CurrentStates.join(allocateStateEvent(CheckerID, To, Location));

      return Result;
    }

    if (CurrentStates.isUninitialized())
      return Result;

    if (const StateEvent *FromEvent =
            CurrentStates.getValue().find(CheckerID, *From)) {
      const StateEvent &ToEvent =
          allocateStateEvent(CheckerID, To, Location, FromEvent);
      StateLatticeValue NewValue =
          CurrentStates.getValue().replace(*FromEvent, ToEvent);
      CurrentStates.getValue() = NewValue;
      Result = mlir::ChangeResult::Change;

      // FIXME: this functionality doesn't belong here
      if (To.isError()) {
        auto &Checker = findChecker(CheckerID);
        InFlightDiagnostic Error = Checker.emitError(Location, To);

        for (const StateEvent *CurrentEvent = ToEvent.Parent; CurrentEvent;
             CurrentEvent = CurrentEvent->Parent) {
          Checker.emitNote(Error, CurrentEvent->Location,
                           CurrentEvent->Key.State);
        }
      }
    }

    return Result;
  }

  AssociatedStates &
  getOperandByIdx(Operation *Op, unsigned Index,
                  ArrayRef<AssociatedStates *> OperandStates) {
    // Index equal to the number of operands means result.
    if (Op->getNumOperands() == Index) {
      assert(Op->getNumResults() && "Operation has no result");
      // OperandStates don't include the result and should be requested
      // separately.
      return getLatticeElement(Op->getResult(0));
    }

    // Otherwise, it's right there in OperandStates.
    return *OperandStates[Index];
  }
};

//===----------------------------------------------------------------------===//
//                                  Interface
//===----------------------------------------------------------------------===//

FlowSensitiveAnalysis::FlowSensitiveAnalysis(mlir::Operation *TopLevelOp) {
  PImpl = std::make_unique<Implementation>(TopLevelOp->getContext());
  PImpl->run(TopLevelOp);
}

FlowSensitiveAnalysis::~FlowSensitiveAnalysis() = default;
