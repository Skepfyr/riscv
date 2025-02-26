#![feature(new_zeroed_alloc)]
#![warn(missing_docs)]
#![allow(clippy::unusual_byte_groupings)]
//! A RISC-V Computer emulator

pub mod assembler;
pub mod traps;

use core::{convert::TryInto, ops::Range};
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
};
use strum::{EnumDiscriminants, FromRepr};
use traps::Exception;

/// A RISC-V computer
///
/// Contains a single [`Core`] and the main [`Memory`].
pub struct Computer {
    /// The emulated processor that the program will run on.
    pub core: Core,
    /// The computer's memory, also contains any DMA peripherals.
    pub memory: Memory,
}

impl Computer {
    /// Creates a new computer with the specified amount of memory.
    ///
    /// All the memory is initialised to zero.
    /// This panics if too much memory is requested.
    pub fn new(memory_size: u64) -> Self {
        Self {
            core: Core::new(),
            memory: Memory::new(memory_size),
        }
    }

    /// Runs the currently loaded program until an [`Exception`] is hit.
    ///
    /// This resumes from whatever state the processor is in, so this can be
    /// called repeatedly after appropriately dealing with the exception.
    ///
    /// This currently ignores any raised [`Exception::EnvironmentCall`].
    pub fn run(&mut self) -> Result<(), Exception> {
        loop {
            match self.core.step(&self.memory) {
                Ok(()) => {
                    if *self.memory.magic.get_mut() != 0 {
                        break;
                    }
                }
                Err(Exception::MModeEnvironmentCall) => {
                    self.core.program_counter += 4;
                    if self.core.registers[10] == 0 {
                        println!("{}", self.core.registers[13]);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// The address that the processor will start executing from after a reset.
pub const RESET_VECTOR: u64 = 0x2000;

/// A systems's memory.
///
/// Currently this only holds main memory but it is intended to also hold
/// (references to) the memory of any attached DMA peripherals.
pub struct Memory {
    /// The ROM memory, holds the initial program and any boot code.
    pub rom: Box<[u8; 0x4000]>,
    /// The magic register, used to shutdown the computer.
    pub magic: AtomicU32,
    /// The main system memory, holds both instructions and data.
    pub main: Mutex<MainMemory>,
}

impl Memory {
    /// The offset in the address space where main memory starts.
    pub const RAM_OFFSET: usize = 0x40_00_00_00;

    /// Constructs a Memory containing the provided data in main memory.
    pub fn new(main_memory_size: u64) -> Self {
        Self {
            rom: unsafe { Box::new_zeroed().assume_init() },
            magic: AtomicU32::new(0),
            main: Mutex::new(MainMemory::new(main_memory_size.try_into().unwrap())),
        }
    }

    fn load(&self, address: u64, width: u8, sign_extend: bool) -> Result<u64, Exception> {
        let num_bytes = 1 << width;
        // If any bits below width aren't set this load isn't correctly aligned.
        if bits(address as u32, 0..width) != 0 {
            return Err(Exception::LoadAddressMisaligned);
        }
        let num_bytes = num_bytes as usize;
        let address: usize = address.try_into().map_err(|_| Exception::LoadAccessFault)?;
        let main;
        let bytes;
        let data = match address {
            0..0x4000 => {
                // Get a slice of ROM from address to address + 2^width.
                self.rom
                    .get(address..address + num_bytes)
                    .ok_or(Exception::LoadAccessFault)?
            }
            0x4000 => {
                bytes = self.magic.load(Ordering::Relaxed).to_le_bytes();
                bytes.get(..num_bytes).ok_or(Exception::LoadAccessFault)?
            }
            0x4001..=0x4004 => {
                // The magic register has extra alignment requirements.
                return Err(Exception::LoadAddressMisaligned);
            }
            Self::RAM_OFFSET.. => {
                // Main memory is offset to allow for ROM and peripherals.
                let address = address - Self::RAM_OFFSET;
                main = self.main.lock().unwrap();
                // Get a slice of main memory from address to address + 2^width.
                main.get(address..address + num_bytes)
                    .ok_or(Exception::LoadAccessFault)?
            }
            _ => return Err(Exception::LoadAccessFault),
        };
        // Sign extend by copying into an array of all ones, but only do it if
        // sign extending has been requested and the top bit of the loaded data
        // is set.
        let mut bytes = if sign_extend && (data[num_bytes - 1] as i8) < 0 {
            [255; 8]
        } else {
            [0; 8]
        };
        bytes[..num_bytes].copy_from_slice(data);
        Ok(u64::from_le_bytes(bytes))
    }

    fn store(&self, address: u64, width: u8, value: u64) -> Result<(), Exception> {
        let num_bytes = 1 << width;
        // If any bits below width aren't set this load isn't correctly aligned.
        if bits(address as u32, 0..width) != 0 {
            return Err(Exception::StoreAddressMisaligned);
        }
        let num_bytes = num_bytes as usize;
        let address: usize = address
            .try_into()
            .map_err(|_| Exception::StoreAccessFault)?;
        match address {
            0x4000 => {
                // The magic register is only 4 bytes wide.
                if num_bytes != 4 {
                    return Err(Exception::StoreAccessFault);
                }
                self.magic.store(value as u32, Ordering::Relaxed);
                return Ok(());
            }
            0x4001..=0x4004 => {
                // The magic register has extra alignment requirements.
                return Err(Exception::StoreAddressMisaligned);
            }
            0x40000000.. => {
                // Main memory is offset by 0x40000000 to allow for ROM and peripherals.
                let address = address - 0x40000000;
                let mut main = self.main.lock().unwrap();
                // Get a slice of main memory from address to address + 2^width.
                let data = main
                    .get_mut(address..address + num_bytes)
                    .ok_or(Exception::StoreAccessFault)?;
                data.copy_from_slice(&value.to_le_bytes()[..num_bytes]);
            }
            _ => return Err(Exception::StoreAccessFault),
        }

        Ok(())
    }
}

/// Represents RAM.
///
/// This is heap allocated and zero-initialized.
pub struct MainMemory {
    ptr: NonNull<u8>,
    len: usize,
}

impl MainMemory {
    const ALIGN: usize = 4096;

    /// Zero-initializes a block of memory of len bytes.
    pub fn new(len: usize) -> Self {
        if len > isize::MAX as usize {
            panic!("Cannot have more than isize::MAX bytes of main memory");
        }
        let ptr = if len > 0 {
            let layout = Layout::from_size_align(len, Self::ALIGN).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) };
            NonNull::new(ptr).unwrap()
        } else {
            NonNull::dangling()
        };
        Self { ptr, len }
    }
}

impl Drop for MainMemory {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align(self.len, Self::ALIGN).unwrap(),
                );
            }
        }
    }
}

impl Deref for MainMemory {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for MainMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

const fn bits(num: u32, range: Range<u8>) -> u32 {
    // Right shift until the first bit is the least significant bit, then AND
    // that with a string of `range.len()` ones to isolate them.
    (num >> range.start) & !(u32::MAX << (range.end - range.start))
}

const fn bit(num: u32, bit: u8) -> bool {
    (num >> bit) & 1 != 0
}

const fn sext(num: u32, bit: u8) -> i32 {
    // Extend the `bit`th bit to the end by shifting it left and letting right
    // shift do the work.
    let inv = 32 - bit;
    ((num as i32) << inv) >> inv
}

/// A processing core, represents a single [RISC-V] hart.
///
/// This implements a slight superset of RV64I, using a single instruction per
/// clock cycle model.
///
/// [RISC-V]: https://riscv.org/specifications/
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Core {
    /// The integer registers, the 0th register should always be 0.
    pub registers: [u64; 32],
    /// The float registers, none of these carry any special meaning.
    pub float_registers: [u64; 32],
    /// The location in memory of the next instruction to execute.
    pub program_counter: u64,
    /// The number of clock cycles this core has completed.
    pub cycle_count: u64,
}

impl Default for Core {
    fn default() -> Self {
        Self {
            registers: [0; 32],
            float_registers: [0; 32],
            program_counter: RESET_VECTOR,
            cycle_count: 0,
        }
    }
}

impl Core {
    /// Create a new core with all registers set to 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the core to its default state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Execute a single clock cycle.
    ///
    /// This returns an error if an [`Exception`] occurs, although these do not
    /// always indicate an actual error has occurred.
    pub fn step(&mut self, mem: &Memory) -> Result<(), Exception> {
        let instruction_value = mem.load(self.program_counter, 2, false)? as u32;
        let instruction = Instruction::try_from(instruction_value)?;
        self.execute(instruction, mem)?;
        self.registers[0] = 0;
        // Update the program counter to point at the next instruction.
        self.program_counter += 4;
        // It makes no difference doing this at the start or end as long as it
        // is consistent.
        self.cycle_count += 1;
        Ok(())
    }

    fn execute(&mut self, inst: Instruction, mem: &Memory) -> Result<(), Exception> {
        match inst {
            Instruction::Load(op) => {
                let value = mem.load(
                    self.registers[op.source_register as usize]
                        .wrapping_add_signed(op.immediate as i64),
                    op.funct3 & 0b11,
                    op.funct3 & 0b100 == 0,
                )?;
                self.registers[op.destination_register as usize] = value;
            }
            Instruction::LoadFp(_op) => return Err(Exception::IllegalInstruction),
            Instruction::MiscMem(_op) => {
                // Currently do nothing for all MiscMem instructions.
                // This is acceptable for FENCE.I and it's the only one we support right now.s
            }
            Instruction::OpImm(op) => {
                self.registers[op.destination_register as usize] = alu(
                    op.funct3,
                    None,
                    false,
                    self.registers[op.source_register as usize],
                    op.immediate as i64 as u64,
                );
            }
            Instruction::AuiPc(op) => {
                self.registers[op.destination_register as usize] = self
                    .program_counter
                    .wrapping_add_signed(op.immediate as i64);
            }
            Instruction::OpImm32(op) => {
                self.registers[op.destination_register as usize] = alu(
                    op.funct3,
                    None,
                    true,
                    self.registers[op.source_register as usize],
                    op.immediate as i64 as u64,
                );
            }
            Instruction::Store(op) => {
                mem.store(
                    self.registers[op.source_register_1 as usize]
                        .wrapping_add_signed(op.immediate as i64),
                    op.funct3 & 0b11,
                    self.registers[op.source_register_2 as usize],
                )?;
                return Ok(());
            }
            Instruction::StoreFp(_op) => return Err(Exception::IllegalInstruction),
            Instruction::Amo(_op) => return Err(Exception::IllegalInstruction),
            Instruction::Op(op) => {
                self.registers[op.destination_register as usize] = alu(
                    op.funct3,
                    Some(op.funct7),
                    false,
                    self.registers[op.source_register_1 as usize],
                    self.registers[op.source_register_2 as usize],
                );
            }
            Instruction::Lui(op) => {
                self.registers[op.destination_register as usize] = op.immediate as u64;
            }
            Instruction::Op32(op) => {
                self.registers[op.destination_register as usize] = alu(
                    op.funct3,
                    Some(op.funct7),
                    true,
                    self.registers[op.source_register_1 as usize],
                    self.registers[op.source_register_2 as usize],
                );
            }
            Instruction::MAdd(_op) => return Err(Exception::IllegalInstruction),
            Instruction::MSub(_op) => return Err(Exception::IllegalInstruction),
            Instruction::NmSub(_op) => return Err(Exception::IllegalInstruction),
            Instruction::NmAdd(_op) => return Err(Exception::IllegalInstruction),
            Instruction::OpFp(_op) => return Err(Exception::IllegalInstruction),
            Instruction::OpV(_op) => return Err(Exception::IllegalInstruction),
            Instruction::Branch(op) => {
                let value1 = self.registers[op.source_register_1 as usize];
                let value2 = self.registers[op.source_register_2 as usize];
                let funct3 = op.funct3 as u32;
                let (not, unsigned, less) = (bit(funct3, 0), bit(funct3, 1), bit(funct3, 2));
                let branch_equal = value1 == value2;
                let branch_less = if unsigned {
                    value1 < value2
                } else {
                    (value1 as i64) < (value2 as i64)
                };
                if not ^ if less { branch_less } else { branch_equal } {
                    // Offset by 4 so we can unconditionally add 4 to the program counter.
                    self.program_counter = self
                        .program_counter
                        .wrapping_add_signed(op.immediate as i64)
                        .wrapping_add_signed(-4);
                }
            }
            Instruction::Jalr(op) => {
                let link_value = self.program_counter.wrapping_add(4);
                self.program_counter = self.registers[op.source_register as usize]
                    .wrapping_add_signed(op.immediate as i64)
                    .wrapping_add_signed(-4);
                self.program_counter &= !1;
                self.registers[op.destination_register as usize] = link_value;
            }
            Instruction::Jal(op) => {
                let link_value = self.program_counter.wrapping_add(4);
                self.program_counter = self
                    .program_counter
                    .wrapping_add_signed(op.immediate as i64)
                    .wrapping_add_signed(-4);
                self.registers[op.destination_register as usize] = link_value;
            }
            Instruction::System(op) => {
                return Err(if bit(op.funct12.into(), 0) {
                    Exception::Breakpoint
                } else {
                    Exception::MModeEnvironmentCall
                })
            }
            Instruction::OpVe(_op) => return Err(Exception::IllegalInstruction),
        }
        Ok(())
    }
}

fn alu(op: u8, variant: Option<u8>, is_32bit: bool, a: u64, b: u64) -> u64 {
    let out = match op {
        0b000 => {
            if variant.is_some_and(|variant| bit(variant as u32, 5)) {
                a.wrapping_sub(b)
            } else {
                a.wrapping_add(b)
            }
        }
        0b001 => {
            if is_32bit {
                (a as u32).wrapping_shl(b as u32) as u64
            } else {
                a.wrapping_shl(b as u32)
            }
        }
        0b101 => {
            let arithmetic = match variant {
                Some(variant) => bit(variant as u32, 5),
                None => bit(b as u32, 10),
            };
            match (is_32bit, arithmetic) {
                (false, false) => a.wrapping_shr(b as u32),
                (false, true) => (a as i64).wrapping_shr(b as u32) as u64,
                (true, false) => (a as i32 as u32).wrapping_shr(b as u32) as u64,
                (true, true) => (a as i32).wrapping_shr(b as u32) as u64,
            }
        }
        0b010 => {
            if (a as i64) < (b as i64) {
                1
            } else {
                0
            }
        }
        0b011 => {
            if a < b {
                1
            } else {
                0
            }
        }
        0b100 => a ^ b,
        0b110 => a | b,
        0b111 => a & b,
        // We know alu_op is three bits
        8..=u8::MAX => unreachable!(),
    };
    // Sign extend the results to 32 bits for the special 32-bit instructions.
    if is_32bit {
        out as i32 as i64 as u64
    } else {
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, EnumDiscriminants)]
#[strum_discriminants(derive(FromRepr))]
#[repr(u8)]
enum Instruction {
    Load(OpImmediate) = 0b00_000_11,
    LoadFp(OpImmediate) = 0b00_001_11,
    // Custom0 = 0b00_010_11,
    MiscMem(OpRegister2) = 0b00_011_11,
    OpImm(OpImmediate) = 0b00_100_11,
    AuiPc(OpUpper) = 0b00_101_11,
    OpImm32(OpImmediate) = 0b00_110_11,
    // 48-bit instructions = 0b00_111_11,
    Store(OpStore) = 0b01_000_11,
    StoreFp(OpStore) = 0b01_001_11,
    // Custom1 = 0b01_010_11,
    Amo(OpRegister) = 0b01_011_11,
    Op(OpRegister) = 0b01_100_11,
    Lui(OpUpper) = 0b01_101_11,
    Op32(OpRegister) = 0b01_110_11,
    // 64-bit instructions = 0b01_111_11,
    MAdd(OpRegister4) = 0b10_000_11,
    MSub(OpRegister4) = 0b10_001_11,
    NmSub(OpRegister4) = 0b10_010_11,
    NmAdd(OpRegister4) = 0b10_011_11,
    OpFp(OpRegister) = 0b10_100_11,
    OpV(OpRegister) = 0b10_101_11,
    // Custom2 = 0b10_110_11,
    // 48-bit instructions = 0b10_111_11,
    Branch(OpBranch) = 0b11_000_11,
    Jalr(OpImmediate) = 0b11_001_11,
    Jal(OpJump) = 0b11_011_11,
    System(OpRegister2) = 0b11_100_11,
    OpVe(OpRegister2) = 0b11_101_11,
    // Custom3 = 0b11_110_11,
    // >=80-bit instructions = 0b11_111_11,
}

impl TryFrom<u32> for Instruction {
    type Error = Exception;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let opcode = InstructionDiscriminants::from_repr(bits(value, 0..7) as u8)
            .ok_or(Exception::IllegalInstruction)?;
        Ok(match opcode {
            InstructionDiscriminants::Load => Instruction::Load(value.into()),
            InstructionDiscriminants::LoadFp => Instruction::LoadFp(value.into()),
            InstructionDiscriminants::MiscMem => Instruction::MiscMem(value.into()),
            InstructionDiscriminants::OpImm => Instruction::OpImm(value.into()),
            InstructionDiscriminants::AuiPc => Instruction::AuiPc(value.into()),
            InstructionDiscriminants::OpImm32 => Instruction::OpImm32(value.into()),
            InstructionDiscriminants::Store => Instruction::Store(value.into()),
            InstructionDiscriminants::StoreFp => Instruction::StoreFp(value.into()),
            InstructionDiscriminants::Amo => Instruction::Amo(value.into()),
            InstructionDiscriminants::Op => Instruction::Op(value.into()),
            InstructionDiscriminants::Lui => Instruction::Lui(value.into()),
            InstructionDiscriminants::Op32 => Instruction::Op32(value.into()),
            InstructionDiscriminants::MAdd => Instruction::MAdd(value.into()),
            InstructionDiscriminants::MSub => Instruction::MSub(value.into()),
            InstructionDiscriminants::NmSub => Instruction::NmSub(value.into()),
            InstructionDiscriminants::NmAdd => Instruction::NmAdd(value.into()),
            InstructionDiscriminants::OpFp => Instruction::OpFp(value.into()),
            InstructionDiscriminants::OpV => Instruction::OpV(value.into()),
            InstructionDiscriminants::Branch => Instruction::Branch(value.into()),
            InstructionDiscriminants::Jalr => Instruction::Jalr(value.into()),
            InstructionDiscriminants::Jal => Instruction::Jal(value.into()),
            InstructionDiscriminants::System => Instruction::System(value.into()),
            InstructionDiscriminants::OpVe => Instruction::OpVe(value.into()),
        })
    }
}

fn rd(inst: u32) -> Register {
    Register::from_repr(bits(inst, 7..12) as u8)
        .expect("5-bits means all valid numbers are valid registers")
}

fn rs1(inst: u32) -> Register {
    Register::from_repr(bits(inst, 15..20) as u8)
        .expect("5-bits means all valid numbers are valid registers")
}

fn rs2(inst: u32) -> Register {
    Register::from_repr(bits(inst, 20..25) as u8)
        .expect("5-bits means all valid numbers are valid registers")
}

fn rs3(inst: u32) -> Register {
    Register::from_repr(bits(inst, 27..32) as u8)
        .expect("5-bits means all valid numbers are valid registers")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpRegister {
    destination_register: Register,
    source_register_1: Register,
    source_register_2: Register,
    funct3: u8,
    funct7: u8,
}

impl From<u32> for OpRegister {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            source_register_1: rs1(value),
            source_register_2: rs2(value),
            funct3: bits(value, 12..15) as u8,
            funct7: bits(value, 25..32) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpRegister2 {
    destination_register: Register,
    source_register: Register,
    funct3: u8,
    funct12: u16,
}

impl From<u32> for OpRegister2 {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            source_register: rs1(value),
            funct3: bits(value, 12..15) as u8,
            funct12: bits(value, 20..32) as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpRegister4 {
    destination_register: Register,
    source_register_1: Register,
    source_register_2: Register,
    source_register_3: Register,
    funct3: u8,
    funct2: u8,
}

impl From<u32> for OpRegister4 {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            source_register_1: rs1(value),
            source_register_2: rs2(value),
            source_register_3: rs3(value),
            funct3: bits(value, 12..15) as u8,
            funct2: bits(value, 25..27) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpImmediate {
    destination_register: Register,
    source_register: Register,
    immediate: i32,
    funct3: u8,
}

impl From<u32> for OpImmediate {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            source_register: rs1(value),
            immediate: sext(bits(value, 20..32), 12),
            funct3: bits(value, 12..15) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpStore {
    source_register_1: Register,
    source_register_2: Register,
    immediate: i32,
    funct3: u8,
}

impl From<u32> for OpStore {
    fn from(value: u32) -> Self {
        Self {
            source_register_1: rs1(value),
            source_register_2: rs2(value),
            immediate: sext((bits(value, 25..32) << 5) | bits(value, 7..12), 12),
            funct3: bits(value, 12..15) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpBranch {
    source_register_1: Register,
    source_register_2: Register,
    immediate: i32,
    funct3: u8,
}

impl From<u32> for OpBranch {
    fn from(value: u32) -> Self {
        Self {
            source_register_1: rs1(value),
            source_register_2: rs2(value),
            immediate: sext(
                (bits(value, 31..32) << 12)
                    | (bits(value, 7..8) << 11)
                    | (bits(value, 25..31) << 5)
                    | (bits(value, 8..12) << 1),
                13,
            ),
            funct3: bits(value, 12..15) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpUpper {
    destination_register: Register,
    immediate: i32,
}

impl From<u32> for OpUpper {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            immediate: sext(bits(value, 12..32) << 12, 32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OpJump {
    destination_register: Register,
    immediate: i32,
}

impl From<u32> for OpJump {
    fn from(value: u32) -> Self {
        Self {
            destination_register: rd(value),
            immediate: sext(
                (bits(value, 31..32) << 20)
                    | (bits(value, 12..20) << 12)
                    | (bits(value, 20..21) << 11)
                    | (bits(value, 21..31) << 1),
                21,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, FromRepr)]
#[repr(u8)]
enum Register {
    X0 = 0,
    X1 = 1,
    X2 = 2,
    X3 = 3,
    X4 = 4,
    X5 = 5,
    X6 = 6,
    X7 = 7,
    X8 = 8,
    X9 = 9,
    X10 = 10,
    X11 = 11,
    X12 = 12,
    X13 = 13,
    X14 = 14,
    X15 = 15,
    X16 = 16,
    X17 = 17,
    X18 = 18,
    X19 = 19,
    X20 = 20,
    X21 = 21,
    X22 = 22,
    X23 = 23,
    X24 = 24,
    X25 = 25,
    X26 = 26,
    X27 = 27,
    X28 = 28,
    X29 = 29,
    X30 = 30,
    X31 = 31,
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Remove this and use Computer
    fn from_instructions(instructions: &[u32]) -> (Core, Memory) {
        let size = instructions.len() * 4;
        let mut memory = Memory::new(size as u64);
        let mut bytes = instructions.iter().flat_map(|i| i.to_le_bytes());
        memory.main.get_mut().unwrap()[..size].fill_with(|| bytes.next().unwrap());
        let mut core = Core::new();
        core.program_counter = 0x40000000;
        (core, memory)
    }

    #[test]
    fn add() {
        let instructions = vec![0b0000000_00001_00010_000_00011_0110011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 1;
        core.registers[2] = 3;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 4);
    }

    #[test]
    fn addi() {
        let instructions = vec![0b010010100001_00010_000_00011_0010011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[2] = 0b1001101;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0b010011101110);
    }

    #[test]
    fn addiw() {
        let instructions = vec![0b010010100001_00010_000_00011_0011011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[2] = 0x5e51163ec4e02f46;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0xffffffffc4e033e7);
    }

    #[test]
    fn sub() {
        let instructions = vec![0b0100000_00010_00001_000_00011_0110011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 0x8000;
        core.registers[2] = (-0x11i64) as u64;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0x8011);
    }

    #[test]
    fn subw() {
        let instructions = vec![0b0100000_00010_00001_000_00011_0111011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 0x8000;
        core.registers[2] = (-0x11i64) as u64;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0x8011);
    }

    #[test]
    fn srai() {
        let instructions = vec![0b010000_100000_00001_101_00011_0010011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 0x8000000000000000;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0xffffffff80000000);
    }

    #[test]
    fn sraiw() {
        let instructions = vec![0b010000_0_00110_00001_101_00011_0011011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 0x5e51163ec4e02f46;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0xffffffffff1380bd);
    }

    #[test]
    fn sllw() {
        let instructions = vec![0b000000_0_00001_00001_001_00011_0111011];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[1] = 0xffff7fffffffffff;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[3], 0xffffffff80000000);
    }

    #[test]
    fn lh() {
        let instructions = vec![0b111111111101_00001_001_00010_0000011];
        let (mut core, mut mem) = from_instructions(&instructions);
        mem.rom[2..4].copy_from_slice(&(-48i16).to_le_bytes());
        core.registers[1] = 5;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[2], -48i64 as u64);
    }

    #[test]
    fn lbu() {
        let instructions = vec![0b111111111101_00001_100_00010_0000011];
        let (mut core, mut mem) = from_instructions(&instructions);
        mem.rom[1] = 0xfc;
        core.registers[1] = 4;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[2], 0xfcu64);
    }

    #[test]
    fn sb() {
        let instructions = vec![0b1111111_00101_00001_000_00010_0100011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 0x40000021;
        core.registers[5] = 42 + (5 << 8);
        core.step(&mem).unwrap();
        assert_eq!(mem.main.get_mut().unwrap()[3], 42);
    }

    #[test]
    fn sd() {
        let instructions = vec![0b1111111_00101_00001_011_00010_0100011, 0];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 0x4000001E;
        core.registers[5] = -42i64 as u64;
        core.step(&mem).unwrap();
        assert_eq!(&mem.main.get_mut().unwrap()[0..8], (-42i64).to_le_bytes());
    }

    #[test]
    fn countdown() {
        let instructions = vec![
            0b000000001010_00000_000_00001_0010011,
            0b111111111111_00001_000_00001_0010011,
            0b1111111_00000_00001_101_11101_1100011,
        ];
        let (mut core, mem) = from_instructions(&instructions);
        while core.program_counter < 0x40000000 + 12 {
            core.step(&mem).unwrap();
        }
        assert_eq!(core.cycle_count, 23);
    }

    #[test]
    fn jal() {
        let instructions = vec![0b00000000011000000000_00001_1101111];
        let (mut core, mem) = from_instructions(&instructions);
        core.step(&mem).unwrap();
        assert_eq!(core.registers[1], 0x40000000 + 4);
        assert_eq!(core.program_counter, 0x40000000 + 6);
    }

    #[test]
    fn jalr() {
        let instructions = vec![0b00000000111_00010_000_00001_1100111];
        let (mut core, mem) = from_instructions(&instructions);
        core.registers[2] = 5;
        core.step(&mem).unwrap();
        assert_eq!(core.registers[1], 0x40000000 + 4);
        assert_eq!(core.program_counter, 12);
    }

    #[test]
    fn lui() {
        let instructions = vec![0b10101010101010101010_00001_0110111];
        let (mut core, mem) = from_instructions(&instructions);
        core.step(&mem).unwrap();
        assert_eq!(core.registers[1], 18446744072277893120);
    }

    #[test]
    fn auipc() {
        let instructions = vec![
            0b10101010101010101010_00001_0010111,
            0b01010101010101010101_00010_0010111,
        ];
        let (mut core, mem) = from_instructions(&instructions);
        core.step(&mem).unwrap();
        core.step(&mem).unwrap();
        assert_eq!(
            core.registers[1],
            (0xffffffff << 32) + 0b11101010101010101010_000000000000
        );
        assert_eq!(core.registers[2], 0b10010101010101010101_000000000100);
    }

    #[test]
    fn fence() {
        let instructions = vec![0b1000_1111_1111_00000_000_00000_0001111];
        let (mut core, mem) = from_instructions(&instructions);
        core.step(&mem).unwrap();
        let (mut new_core, _new_mem) = from_instructions(&instructions);
        new_core.program_counter += 4;
        new_core.cycle_count += 1;
        assert_eq!(core, new_core);
    }

    #[test]
    fn env_call() {
        let instructions = vec![0b000000000000_00000_000_00000_1110011];
        let (mut core, mem) = from_instructions(&instructions);
        let res = core.step(&mem);
        assert_eq!(res, Err(Exception::MModeEnvironmentCall));
    }

    #[test]
    fn breakpoint() {
        let instructions = vec![0b000000000001_00000_000_00000_1110011];
        let (mut core, mem) = from_instructions(&instructions);
        let res = core.step(&mem);
        assert_eq!(res, Err(Exception::Breakpoint));
    }
}
