#include "tau/Core/Events.h"
#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Operation.h>

namespace tau::core {

LinearChainOfEvents EventHierarchy::linearizeChainOfEvents(
    const AbstractEvent &Event,
    const TopoOrderBlockEnumerator &Enumerator) const {
  LinearChainOfEvents Result;
  llvm::DenseSet<const void *> Visited;

  // Helper function to recursively collect events
  std::function<void(const AbstractEvent &)> CollectEvents =
      [&](const AbstractEvent &CurrentEvent) {
        if (Visited.insert(CurrentEvent.getOpaqueValue()).second) {
          if (const auto *StateEvt =
                  CurrentEvent.dyn_cast<const StateEvent *>()) {
            for (const auto &Parent : StateEvt->getParents()) {
              CollectEvents(Parent);
            }
          } else if (const auto *DataFlowEvt =
                         CurrentEvent.dyn_cast<const DataFlowEvent *>()) {
            for (const auto &Parent : DataFlowEvt->getParents()) {
              CollectEvents(Parent);
            }
          }
          Result.push_back(CurrentEvent);
        }
      };

  // Collect all events
  CollectEvents(Event);

  // Sort the events
  llvm::sort(Result, [&Enumerator](const AbstractEvent &A,
                                   const AbstractEvent &B) {
    mlir::Operation *OpA = A.is<const StateEvent *>()
                               ? A.get<const StateEvent *>()->getLocation()
                               : A.get<const DataFlowEvent *>()->getLocation();

    mlir::Operation *OpB = B.is<const StateEvent *>()
                               ? B.get<const StateEvent *>()->getLocation()
                               : B.get<const DataFlowEvent *>()->getLocation();

    mlir::Block *BlockA = OpA->getBlock();
    mlir::Block *BlockB = OpB->getBlock();

    if (BlockA != BlockB) {
      return Enumerator.getTopoOrderIndex(BlockA) <
             Enumerator.getTopoOrderIndex(BlockB);
    }

    // If in the same block, compare their positions within the
    // block
    return OpB->isBeforeInBlock(OpA);
  });

  return Result;
}

} // end namespace tau::core
