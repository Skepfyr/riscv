//! Traps - Exceptions and Interrupts
//!
//! This module holds the types for all exceptions and interrupts that can be
//! raised by a running [`Core`][super::Core].

/// An enumeration of all the exceptions that could occur
///
/// This is a mixture be Contained, Requested, Invisible, and Fatal traps, these
/// may need to be separated out in the future.
///
/// Not all of these are currently used, but it is intended that they will be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u64)]
pub enum Exception {
    /// Attempted to load an instruction that is incorrectly aligned.
    InstructionAddressMisaligned = 0x0,
    /// Attempted to load an instruction from a region that cannot be accessed.
    InstructionAccessFault = 0x1,
    /// Attempted to execute an illegal instruction. Usually this is because
    /// the instruction doesn't exist.
    IllegalInstruction = 0x2,
    /// A debug breakpoint SYSTEM instruction was executed.
    Breakpoint = 0x3,
    /// Attempted to load from memory at an misaligned address.
    LoadAddressMisaligned = 0x4,
    /// Attempted to load a memory region that cannot be. e.g. for alignment,
    /// permissions, or existence.
    LoadAccessFault = 0x5,
    /// Attempted to store to memory at an misaligned address.
    StoreAddressMisaligned = 0x6,
    /// Attempted to store to a memory region that cannot be. e.g. for alignment,
    /// permissions, or existence.
    StoreAccessFault = 0x7,

    /// A environment call SYSTEM instruction was executed from U-mode.
    UModeEnvironmentCall = 0x8,
    /// A environment call SYSTEM instruction was executed from S-mode.
    SModeEnvironmentCall = 0x9,
    /// A environment call SYSTEM instruction was executed from M-mode.
    MModeEnvironmentCall = 0xB,

    /// An instruction page fault occurred during an instruction fetch.
    InstructionPageFault = 0xC,
    /// A load page fault occurred during a load operation.
    LoadPageFault = 0xD,
    /// A store page fault occurred during a store operation.
    StorePageFault = 0xF,

    /// Triggered by a violation of a check or assertion defined by an ISA
    /// extension that aim to safeguard the integrity of software assets,
    /// including e.g. control-flow and memory-access constraints.
    SoftwareCheck = 0x12,
    /// Triggered when corrupted or uncorrectable data is accessed explicitly or
    /// implicitly by an instruction.
    HardwareError = 0x13,

    
}
