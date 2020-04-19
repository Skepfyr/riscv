#![warn(missing_docs)]
#![allow(clippy::inconsistent_digit_grouping, clippy::unreadable_literal)]
#![feature(exclusive_range_pattern, array_value_iter, or_patterns)]
//! A RISC-V Computer emulator

pub mod assembler;
pub mod traps;

use core::{convert::TryInto, ops::Range};
use traps::Exception;

/// A RISC-V computer
///
/// Contains a single [Core] and the main [Memory].
///
/// [Core]: ./struct.Core.html
/// [Memory]: ./struct.Memory.html
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        let main_mem = vec![0; memory_size.try_into().unwrap()];
        Self {
            core: Core::new(),
            memory: Memory::new(main_mem.into_boxed_slice()),
        }
    }

    /// Runs the currently loaded program until an [Exception] is hit.
    ///
    /// This resumes from whatever state the processor is in, so this can be
    /// called repeatedly after appropriately dealing with the exception.
    ///
    /// This currently ignores any raised [EnvironmentCall].
    ///
    /// [Exception]: ./traps/enum.Exception.html
    /// [EnvironmentCall]: ./traps/enum.Exception.html#variant.EnvironmentCall
    pub fn run(&mut self) -> Exception {
        loop {
            match self.core.step(&mut self.memory) {
                Ok(()) => continue,
                Err(Exception::EnvironmentCall) => {
                    self.core.program_counter += 4;
                    match self.core.registers[10] {
                        0 => println!("{}", self.core.registers[13]),
                        _ => {}
                    }
                    continue;
                }
                Err(e) => return e,
            }
        }
    }

    /// Load a program into memory and point the program counter at the start.
    ///
    /// This copies the provided instructions into the bottom of system memory
    /// and resets the program counter of the core so that it will be executed
    /// when `run` is next called.
    pub fn load_program(&mut self, program: impl IntoIterator<Item = u32>) {
        let bytes: Vec<_> = program
            .into_iter()
            .map(|i| core::array::IntoIter::new(i.to_le_bytes()))
            .flatten()
            .collect();
        self.memory.main[..bytes.len()].copy_from_slice(&bytes);
        self.core.program_counter = 0;
    }
}

/// A systems's memory.
///
/// Currently this only holds main memory but it is intended to also hold
/// (references to) the memory of any attached DMA peripherals.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Memory {
    /// The main system memory, holds both instructions and data.
    pub main: Box<[u8]>,
}

impl Memory {
    /// Constructs a Memory containing the provided data in main memory.
    pub fn new(main: Box<[u8]>) -> Self {
        Self { main }
    }

    fn load(&self, address: u64, width: u8, sign_extend: bool) -> Result<u64, Exception> {
        let num_bytes = 1 << width;
        // If any bits below width aren't set this load isn't correctly aligned.
        if bits(address as u32, 0..width) != 0 {
            return Err(Exception::AccessFault);
        }
        let num_bytes = num_bytes as usize;
        let address: usize = address.try_into().map_err(|_| Exception::AccessFault)?;
        // Get a slice of main memory from address to address + 2^width.
        let data = self
            .main
            .get(address..address + num_bytes)
            .ok_or(Exception::AccessFault)?;
        // Sign extend by copying into an array of all ones, but only do it if
        // sign extending has been requested and the top bit of the loaded data
        // is set.
        let mut bytes = if sign_extend && (data[num_bytes - 1] as i8) < 0 {
            [255; 8]
        } else {
            [0; 8]
        };
        (&mut bytes[..num_bytes]).copy_from_slice(data);
        Ok(u64::from_le_bytes(bytes))
    }

    fn store(&mut self, address: u64, width: u8, value: u64) -> Result<(), Exception> {
        let num_bytes = 1 << width;
        // If any bits below width aren't set this load isn't correctly aligned.
        if bits(address as u32, 0..width) != 0 {
            return Err(Exception::AccessFault);
        }
        let num_bytes = num_bytes as usize;
        let address: usize = address.try_into().map_err(|_| Exception::AccessFault)?;
        // Get a slice of main memory from address to address + 2^width.
        let data = self
            .main
            .get_mut(address..address + num_bytes)
            .ok_or(Exception::AccessFault)?;
        data.copy_from_slice(&value.to_le_bytes()[0..num_bytes]);
        Ok(())
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

const fn sext(num: u32, bit: u8) -> u32 {
    // Extend the `bit`th bit to the end by shifting it left and letting right
    // shift do the work.
    let inv = 32 - bit;
    (((num as i32) << inv) >> inv) as u32
}

/// A processing core, represents a single [RISC-V] hart.
///
/// This implements a slight superset of RV64I, using a single instruction per
/// clock cycle model.
///
/// [RISC-V]: https://riscv.org/specifications/
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Core {
    /// The integer registers, the 0th register shoul always be 0.
    pub registers: [u64; 32],
    /// The float registers, none of these carry any special meaning.
    pub float_registers: [u64; 32],
    /// The location in memory of the next instruction to execute.
    pub program_counter: u64,
    /// The number of clock cycles this core has completed.
    pub cycle_count: u64,
}

impl Core {
    /// Create a new core with all registers set to 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute a single clock cycle.
    ///
    /// This returns an error if any [Exception]s occur, although these do not
    /// always indicate an actual error has occurred.
    ///
    /// [Exception]: ./traps/enum.Exception.html
    pub fn step(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        // It makes no difference doing this at the start or end as long as it
        // is consistent.
        self.cycle_count += 1;
        let fetch = Self::fetch(self.program_counter, mem)?;
        let decode = Self::decode(fetch, &mut self.registers)?;
        let execute = Self::execute(decode);
        // Update the program counter to point at the next instruction, if we
        // should branch take the ALU's output, otherwise just add 4.
        self.program_counter = if execute.branch {
            execute.alu_out
        } else {
            self.program_counter + 4
        };
        let memory = Self::memory(execute, mem)?;
        let dest_register = bits(memory.instruction, 7..12) as usize;
        // If the destination register is the zero register, don't write to it.
        // Otherwise write back for all non-S-or-B-type instructions.
        if dest_register != 0
            && !(bit(execute.instruction, 5) && bits(execute.instruction, 2..5) == 0)
        {
            self.registers[dest_register] = memory.data;
        }
        Ok(())
    }

    fn fetch(program_counter: u64, mem: &Memory) -> Result<FetchStage, Exception> {
        // Just load the next instruction from memory
        Ok(FetchStage {
            program_counter,
            instruction: mem.load(program_counter, 2, false)? as u32,
        })
    }

    fn decode(fetch: FetchStage, registers: &mut [u64; 32]) -> Result<DecodeStage, Exception> {
        let inst = fetch.instruction;
        // If we're a SYSTEM instruction then just immediately raise an exception.
        if bits(inst, 2..7) == 0b11100 {
            return Err(if bit(inst, 20) {
                Exception::Breakpoint
            } else {
                Exception::EnvironmentCall
            });
        }
        // Work out what kind of instruction we have, decode the immediate, and
        // work out how many source registers are needed.
        let (immediate, num_sources) = match (bits(inst, 5..7), bits(inst, 2..5)) {
            // These aren't possible as are outside the bit ranges.
            (0b100..=u32::MAX, _) | (_, 0b1000..=u32::MAX) => unreachable!(),
            // U type instruction - LUI or AUIPC
            (_, 0b101) => (bits(inst, 12..32) << 12, 0),
            // S type instruction - Used for store instructions
            (0b01, 0b000 | 0b001) => (sext(bits(inst, 25..32) << 5 | bits(inst, 7..12), 12), 2),
            // R4 type instruction - Used for floating point instructions
            (0b10, lower) if lower < 4 => (0, 3),
            // R type instruction - Register-Register operations
            (0b01 | 0b10, _) => (0, 2),
            // B type instruction - Used for branch instructions
            (0b11, 0b000) => (
                sext(
                    bits(inst, 31..32) << 12
                        | bits(inst, 7..8) << 11
                        | bits(inst, 25..31) << 5
                        | bits(inst, 8..12) << 1,
                    13,
                ),
                2,
            ),
            // J type instruction - JAL
            (0b11, 0b011) => (
                sext(
                    bits(inst, 31..32) << 20
                        | bits(inst, 12..20) << 12
                        | bits(inst, 20..21) << 11
                        | bits(inst, 21..31) << 1,
                    21,
                ),
                0,
            ),
            // I type instruction - Register-Immediate operations
            (0b00 | 0b11, _) => (sext(bits(inst, 20..32), 12), 1),
        };
        // Utilise the fact that register 0 is always 0.
        let rs1 = if num_sources >= 1 {
            bits(inst, 15..20) as usize
        } else {
            0
        };
        let rs2 = if num_sources >= 2 {
            bits(inst, 20..25) as usize
        } else {
            0
        };
        let rs3 = if num_sources >= 3 {
            bits(inst, 27..32) as usize
        } else {
            0
        };
        Ok(DecodeStage {
            program_counter: fetch.program_counter,
            instruction: inst,
            source1_value: registers[rs1],
            source2_value: registers[rs2],
            source3_value: registers[rs3],
            immediate,
        })
    }

    fn execute(decode: DecodeStage) -> ExecuteStage {
        let inst = decode.instruction;
        let op_upper = bits(inst, 5..7);
        let op_lower = bits(inst, 2..5);
        // Use the program counter fas the base for BRANCH, JAL and AUIPC
        let source1_value = if op_upper == 0b11 && (op_lower == 0b000 || op_lower == 0b011)
            || op_upper == 0b00 && op_lower == 0b101
        {
            decode.program_counter
        } else {
            decode.source1_value
        };
        // Register or immediate depending on the instruction type
        let source2_value = if op_upper == 0b10
            || op_upper == 0b01 && (bit(inst, 3) || bit(inst, 4) && !bit(inst, 2))
        {
            decode.source2_value
        } else {
            decode.immediate as i32 as u64
        };
        // Is this an arithmetic op, otherwise we'll always want to use add.
        let arithmetic_op = !bit(inst, 6) && bit(inst, 4) && !bit(inst, 2);
        let opcode = bits(inst, 2..7) as u8;
        let alu_op = if arithmetic_op {
            bits(inst, 12..15) as u8
        } else {
            // Add if it's not an arithmetic operation.
            0
        };
        // Run alternate versions of the operations in very specific cases.
        let alu_sel = bit(inst, 30) && (opcode == 0b01100 || opcode == 0b00100 && alu_op == 0b101);
        let alu_out = match alu_op {
            0b000 => {
                if alu_sel {
                    source1_value.wrapping_sub(source2_value)
                } else {
                    source1_value.wrapping_add(source2_value)
                }
            }
            0b001 => source1_value.wrapping_shl(source2_value as u32),
            0b101 => {
                if alu_sel {
                    (source1_value as i64).wrapping_shr(source2_value as u32) as u64
                } else {
                    source1_value.wrapping_shr(source2_value as u32)
                }
            }
            0b010 => {
                if (source1_value as i64) < (source2_value as i64) {
                    1
                } else {
                    0
                }
            }
            0b011 => {
                if source1_value < source2_value {
                    1
                } else {
                    0
                }
            }
            0b100 => source1_value ^ source2_value,
            0b110 => source1_value | source2_value,
            0b111 => source1_value & source2_value,
            // We know alu_op is three bits
            8..=u8::MAX => unreachable!(),
        };

        // Work out whether we should be branching.
        let (not, unsigned, less) = (bit(inst, 12), bit(inst, 13), bit(inst, 14));
        let branch_equal = decode.source1_value == decode.source2_value;
        let branch_less = if unsigned {
            decode.source1_value < decode.source2_value
        } else {
            (decode.source1_value as i64) < (decode.source2_value as i64)
        };
        let branch = if bits(inst, 2..7) == 0b11000 {
            not ^ if less { branch_less } else { branch_equal }
        } else {
            bits(inst, 4..7) == 0b110 && bit(inst, 2)
        };

        ExecuteStage {
            program_counter: decode.program_counter,
            instruction: inst,
            alu_out,
            data: decode.source2_value,
            branch,
        }
    }

    fn memory(execute: ExecuteStage, mem: &mut Memory) -> Result<MemoryStage, Exception> {
        // Read from and write to memory where appropriate.
        let op_lower = bits(execute.instruction, 2..5);
        let op_upper = bits(execute.instruction, 5..7);
        let width = bits(execute.instruction, 12..14) as u8;
        let data = if op_upper == 0b00 && op_lower & 0b110 == 0 {
            // A load instruction.
            mem.load(execute.alu_out, width, !bit(execute.instruction, 14))?
        } else if op_upper == 0b01 && op_lower & 0b110 == 0 {
            // A store instruction.
            mem.store(execute.alu_out, width, execute.data)?;
            execute.alu_out
        } else if op_upper == 0b11 && op_lower & 0b101 == 0b001 {
            // Special case write-back for JAL and JALR.
            execute.program_counter + 4
        } else {
            // Everything else just writes the ALU output to a register.
            execute.alu_out
        };
        Ok(MemoryStage {
            instruction: execute.instruction,
            data,
        })
    }
}

/// The registers to hold the output of the fetch pipeline stage
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FetchStage {
    program_counter: u64,
    instruction: u32,
}

/// The registers to hold the output of the decode pipeline stage
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DecodeStage {
    program_counter: u64,
    instruction: u32,
    source1_value: u64,
    source2_value: u64,
    source3_value: u64,
    immediate: u32,
}

/// The registers to hold the output of the execute pipeline stage
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ExecuteStage {
    program_counter: u64,
    instruction: u32,
    alu_out: u64,
    data: u64,
    branch: bool,
}

/// The registers to hold the output of the memory pipeline stage
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MemoryStage {
    instruction: u32,
    data: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::array;

    // TODO: Remove this and use Computer
    fn from_instructions(instructions: &[u32]) -> (Core, Memory) {
        let bytes: Vec<_> = instructions
            .iter()
            .map(|i| array::IntoIter::new(i.to_le_bytes()))
            .flatten()
            .collect();
        (Core::new(), Memory::new(bytes.into_boxed_slice()))
    }

    #[test]
    fn add() {
        let instructions = vec![0b0000000_00001_00010_000_00011_0110011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 1;
        core.registers[2] = 3;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[3], 4);
    }

    #[test]
    fn addi() {
        let instructions = vec![0b010010100001_00010_000_00011_0010011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[2] = 0b1001101;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[3], 0b010011101110);
    }

    #[test]
    fn srai() {
        let instructions = vec![0b010000_110110_00001_101_00011_0010011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = (-2i64 << 0b110110) as u64;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[3], -2i64 as u64);
    }

    #[test]
    #[should_panic] // TODO: This needs fixing, but at the moment this shouldn't work
    fn sraiw() {
        let instructions = vec![0b0100001_10110_00001_101_00011_0010011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = (-2i64 << 0b010110) as u64;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[3], -2i64 as u64);
    }

    #[test]
    fn lh() {
        let instructions = vec![0b111111111101_00001_001_00010_0000011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 5;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[2], -48i64 as u64);
    }

    #[test]
    fn lbu() {
        let instructions = vec![0b111111111101_00001_100_00010_0000011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 4;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[2], 0b11000001u64);
    }

    #[test]
    fn sb() {
        let instructions = vec![0b1111111_00101_00001_000_00010_0100011];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 33;
        core.registers[5] = 42 + (5 << 8);
        core.step(&mut mem).unwrap();
        assert_eq!(mem.main[3], 42);
    }

    #[test]
    fn sd() {
        let instructions = vec![0b1111111_00101_00001_011_00010_0100011, 0];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[1] = 30;
        core.registers[5] = -42i64 as u64;
        core.step(&mut mem).unwrap();
        assert_eq!(&mem.main[0..8], (-42i64).to_le_bytes());
    }

    #[test]
    fn countdown() {
        let instructions = vec![
            0b000000001010_00000_000_00001_0010011,
            0b111111111111_00001_000_00001_0010011,
            0b1111111_00000_00001_101_11101_1100011,
        ];
        let (mut core, mut mem) = from_instructions(&instructions);
        while core.program_counter < 12 {
            core.step(&mut mem).unwrap();
        }
        assert_eq!(core.cycle_count, 23);
    }

    #[test]
    fn jal() {
        let instructions = vec![0b00000000011000000000_00001_1101111];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[1], 4);
        assert_eq!(core.program_counter, 6);
    }

    #[test]
    fn jalr() {
        let instructions = vec![0b00000000111_00010_000_00001_1100111];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.registers[2] = 5;
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[1], 4);
        assert_eq!(core.program_counter, 12);
    }

    #[test]
    fn lui() {
        let instructions = vec![0b10101010101010101010_00001_0110111];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[1], 18446744072277893120);
    }

    #[test]
    fn auipc() {
        let instructions = vec![
            0b10101010101010101010_00001_0010111,
            0b01010101010101010101_00010_0010111,
        ];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.step(&mut mem).unwrap();
        core.step(&mut mem).unwrap();
        assert_eq!(core.registers[1], 18446744072277893120);
        assert_eq!(core.registers[2], 1431654404);
    }

    #[test]
    fn fence() {
        let instructions = vec![0b1000_1111_1111_00000_000_00000_0001111];
        let (mut core, mut mem) = from_instructions(&instructions);
        core.step(&mut mem).unwrap();
        let (mut new_core, new_mem) = from_instructions(&instructions);
        new_core.program_counter = 4;
        new_core.cycle_count = 1;
        assert_eq!(core, new_core);
        assert_eq!(mem, new_mem);
    }

    #[test]
    fn env_call() {
        let instructions = vec![0b000000000000_00000_000_00000_1110011];
        let (mut core, mut mem) = from_instructions(&instructions);
        let res = core.step(&mut mem);
        assert_eq!(res, Err(Exception::EnvironmentCall));
    }

    #[test]
    fn breakpoint() {
        let instructions = vec![0b000000000001_00000_000_00000_1110011];
        let (mut core, mut mem) = from_instructions(&instructions);
        let res = core.step(&mut mem);
        assert_eq!(res, Err(Exception::Breakpoint));
    }
}
