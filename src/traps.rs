//! Traps - Exceptions and Interrupts
//! 
//! This module holds the types for all exceptions and interrupts that can be
//! raised by a running [Core].
//! 
//! [Core]: ../struct.Core.html

/// An enumeration of all the exceptions that could occur
/// 
/// This is a mixture be Contained, Requested, Invisible and Fatal traps, these
/// may need to be separated out in the future.
/// 
/// Not all of these are currently used, but it is intended that they will be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Exception {
    /// Attempted to access a memory region that cannot be. e.g. for alignment,
    /// permissions or existence.
    AccessFault,
    /// Attempted to execute an illegal instruction. Usually this is because
    /// the instruction doesn't exist.
    IllegalInstruction,
    /// Attempted to load an instruction that is incorrectly aligned.
    InstructionAddressMisaligned,
    /// Attempted to load from or store to memory at an misaligned address.
    AddressMisaligned,

    /// A debug breakpoint SYSTEM instruction was executed.
    Breakpoint,
    /// A environment call SYSTEM instruction was executed.
    EnvironmentCall,
}
