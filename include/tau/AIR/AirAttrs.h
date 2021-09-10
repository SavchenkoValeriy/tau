//===- AirAttrs.h - Air attributes ------------------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/AttributeSupport.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/MLIRContext.h>
#include <tuple>

namespace tau {
namespace air {

using StateID = unsigned;

namespace detail {
struct StateChangeAttrStorage : public mlir::AttributeStorage {
  using KeyTy = std::tuple<llvm::StringRef, unsigned, StateID, StateID>;

  StateChangeAttrStorage(const KeyTy &Key) : Key(Key) {}
  StateChangeAttrStorage(llvm::StringRef CheckerID, unsigned Operand,
                         StateID From, StateID To)
      : Key(CheckerID, Operand, From, To) {}

  static StateChangeAttrStorage *
  construct(mlir::AttributeStorageAllocator &Allocator, const KeyTy &Key) {
    return new (Allocator.allocate<StateChangeAttrStorage>())
        StateChangeAttrStorage(Key);
  }

  bool operator==(const KeyTy &Other) const { return Key == Other; }

  KeyTy Key;
};
} // end namespace detail

class StateChangeAttr
    : public mlir::Attribute::AttrBase<StateChangeAttr, mlir::Attribute,
                                       detail::StateChangeAttrStorage> {
public:
  /// Inherit base constructors.
  using Base::Base;
  using Base::getChecked;

  static StateChangeAttr get(mlir::MLIRContext *Context,
                             llvm::StringRef CheckerID, unsigned OperandIdx,
                             StateID From, StateID To);
  static StateChangeAttr get(mlir::MLIRContext *Context,
                             llvm::StringRef CheckerID, unsigned OperandIdx,
                             StateID To);

  llvm::StringRef getCheckerID() const;
  unsigned getOperandIdx() const;
  llvm::Optional<StateID> getFromState() const;
  StateID getToState() const;
};

namespace detail {
struct StateTransferAttrStorage : public mlir::AttributeStorage {
  using KeyTy = std::tuple<llvm::StringRef, unsigned, unsigned, StateID>;

  StateTransferAttrStorage(const KeyTy &Key) : Key(Key) {}
  StateTransferAttrStorage(llvm::StringRef CheckerID, unsigned From,
                           unsigned To, StateID LimitedTo)
      : Key(CheckerID, From, To, LimitedTo) {}
  StateTransferAttrStorage(llvm::StringRef CheckerID, unsigned From,
                           unsigned To)
      : Key(CheckerID, From, To, 0) {}

  static StateTransferAttrStorage *
  construct(mlir::AttributeStorageAllocator &Allocator, const KeyTy &Key) {
    return new (Allocator.allocate<StateTransferAttrStorage>())
        StateTransferAttrStorage(Key);
  }

  bool operator==(const KeyTy &Other) const { return Key == Other; }

  KeyTy Key;
};
} // end namespace detail

class StateTransferAttr
    : public mlir::Attribute::AttrBase<StateTransferAttr, mlir::Attribute,
                                       detail::StateTransferAttrStorage> {
public:
  /// Inherit base constructors.
  using Base::Base;
  using Base::getChecked;

  static StateTransferAttr get(mlir::MLIRContext *Context,
                               llvm::StringRef CheckerID,
                               unsigned FromOperandIdx, unsigned ToOperandIdx);
  static StateTransferAttr get(mlir::MLIRContext *Context,
                               llvm::StringRef CheckerID,
                               unsigned FromOperandIdx, unsigned ToOperandIdx,
                               StateID LimitedTo);

  llvm::StringRef getCheckerID() const;
  unsigned getFromOperandIdx() const;
  unsigned getToOperandIdx() const;
  llvm::Optional<StateID> getLimitingState() const;
};
} // end namespace air
} // end namespace tau
