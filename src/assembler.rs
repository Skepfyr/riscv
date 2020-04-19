//! A RISC-V assembler
//!
//! The main entrypoint to this module is the [assemble] function that takes a
//! the input RISC-V assembler code and produces the program as machine code.
//!
//! [assemble]: ./fn.assemble.html

use crate::bits;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    ops::Range,
};

/// Assembles the input into machine code.
pub fn assemble(input: &str) -> Result<Vec<u32>, String> {
    let (
        _,
        Program {
            instructions,
            labels,
        },
    ) = program(input)?;
    Ok(instructions
        .into_iter()
        .enumerate()
        .map(|(loc, i)| i.to_machine_code(loc * 4, &labels))
        .collect::<Result<_, _>>()?)
}

type ParseResult<'a, T> = Result<(&'a str, T), String>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Program<'a> {
    instructions: Vec<Instruction<'a>>,
    labels: HashMap<&'a str, usize>,
}

#[allow(dead_code)] // Float ops not implemented yet
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Instruction<'a> {
    RType {
        opcode: u32,
        funct3: u32,
        funct7: u32,
        destination: Register,
        source1: Register,
        source2: Register,
    },
    R4Type {
        opcode: u32,
        funct3: u32,
        funct2: u32,
        destination: Register,
        source1: Register,
        source2: Register,
        source3: Register,
    },
    IType {
        opcode: u32,
        funct3: u32,
        destination: Register,
        source1: Register,
        immediate: Immediate<'a, I12>,
    },
    SType {
        opcode: u32,
        funct3: u32,
        source1: Register,
        source2: Register,
        immediate: Immediate<'a, I12>,
    },
    BType {
        opcode: u32,
        funct3: u32,
        source1: Register,
        source2: Register,
        immediate: Immediate<'a, I13_2>,
    },
    UType {
        opcode: u32,
        destination: Register,
        immediate: Immediate<'a, I20>,
    },
    JType {
        opcode: u32,
        destination: Register,
        immediate: Immediate<'a, I21_2>,
    },
    Raw {
        instruction: u32,
    },
}

impl<'a> Instruction<'a> {
    fn to_machine_code(&self, loc: usize, labels: &HashMap<&'a str, usize>) -> Result<u32, String> {
        Ok(match *self {
            Instruction::RType {
                opcode,
                funct3,
                funct7,
                destination,
                source1,
                source2,
            } => {
                funct7 << 25
                    | source2.0 << 20
                    | source1.0 << 15
                    | funct3 << 12
                    | destination.0 << 7
                    | opcode
            }
            Instruction::R4Type {
                opcode,
                funct3,
                funct2,
                destination,
                source1,
                source2,
                source3,
            } => {
                source3.0 << 27
                    | funct2 << 25
                    | source2.0 << 20
                    | source1.0 << 15
                    | funct3 << 12
                    | destination.0 << 7
                    | opcode
            }
            Instruction::IType {
                opcode,
                funct3,
                destination,
                source1,
                immediate,
            } => {
                (immediate.to_offset(loc, labels)? as u32) << 20
                    | source1.0 << 15
                    | funct3 << 12
                    | destination.0 << 7
                    | opcode
            }
            Instruction::SType {
                opcode,
                funct3,
                source1,
                source2,
                immediate,
            } => {
                let imm = immediate.to_offset(loc, labels)? as u32;
                bits(imm, 5..12) << 25
                    | source2.0 << 20
                    | source1.0 << 15
                    | funct3 << 12
                    | bits(imm, 0..5) << 7
                    | opcode
            }
            Instruction::BType {
                opcode,
                funct3,
                source1,
                source2,
                immediate,
            } => {
                let imm = immediate.to_offset(loc, labels)? as u32;
                bits(imm, 12..13) << 31
                    | bits(imm, 5..11) << 25
                    | source2.0 << 20
                    | source1.0 << 15
                    | funct3 << 12
                    | bits(imm, 1..5) << 8
                    | bits(imm, 11..12) << 7
                    | opcode
            }
            Instruction::UType {
                opcode,
                destination,
                immediate,
            } => (immediate.to_offset(loc, labels)? as u32) << 12 | destination.0 << 7 | opcode,
            Instruction::JType {
                opcode,
                destination,
                immediate,
            } => {
                let imm = immediate.to_offset(loc, labels)? as u32;
                bits(imm, 20..21) << 31
                    | bits(imm, 1..11) << 21
                    | bits(imm, 11..12) << 20
                    | bits(imm, 12..20) << 12
                    | destination.0 << 7
                    | opcode
            }
            Instruction::Raw { instruction } => instruction,
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct I12(i32);
impl<'a> TryFrom<i32> for I12 {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let max = 1 << (12 - 1);
        if (-max..max).contains(&value) {
            Ok(I12(value))
        } else {
            Err(format!("{} outside valid immediate range", value))
        }
    }
}
impl Into<i32> for I12 {
    fn into(self) -> i32 {
        self.0
    }
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct I13_2(i32);
impl<'a> TryFrom<i32> for I13_2 {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let max = 1 << (13 - 1);
        if (-max..max).contains(&value) {
            Ok(I13_2(value))
        } else {
            Err(format!("{} outside valid immediate range", value))
        }
    }
}
impl Into<i32> for I13_2 {
    fn into(self) -> i32 {
        self.0
    }
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct I20(i32);
impl<'a> TryFrom<i32> for I20 {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let max = 1 << (20 - 1);
        if (-max..max).contains(&value) {
            Ok(I20(value))
        } else {
            Err(format!("{} outside valid immediate range", value))
        }
    }
}
impl Into<i32> for I20 {
    fn into(self) -> i32 {
        self.0
    }
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct I21_2(i32);
impl TryFrom<i32> for I21_2 {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let max = 1 << (21 - 1);
        if (-max..max).contains(&value) {
            Ok(I21_2(value))
        } else {
            Err(format!("{} outside valid immediate range", value))
        }
    }
}
impl Into<i32> for I21_2 {
    fn into(self) -> i32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Register(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Immediate<'a, T> {
    Label(&'a str),
    Immediate(T),
}

impl<T: Into<i32> + Copy> Immediate<'_, T> {
    fn to_offset(&self, loc: usize, labels: &HashMap<&str, usize>) -> Result<i32, String> {
        Ok(match self {
            Immediate::Label(l) => {
                let dest = *labels.get(l).ok_or(format!("Label {} not defined.", l))?;
                if let Some(diff) = dest.checked_sub(loc) {
                    diff.try_into()
                } else {
                    // Can never overflow
                    let diff = loc.wrapping_sub(dest);
                    diff.try_into().map(|i: i32| -i)
                }
                .map_err(|_| format!("Distance to {} too large", l))?
            }
            Immediate::Immediate(value) => (*value).into(),
        })
    }
}

fn is_space(c: char) -> bool {
    c == ' ' || c == '\t'
}

fn arg_sep(c: char) -> bool {
    c == ',' || is_space(c)
}

fn signed_int(i: &str) -> ParseResult<i32> {
    let loc = i
        .find(|c: char| !c.is_ascii_digit() && c != '+' && c != '-')
        .unwrap_or_else(|| i.len());
    let (num, i) = i.split_at(loc);
    let num = num
        .parse()
        .map_err(|_| format!("\"{}\" is not a valid number", num))?;
    Ok((i, num))
}

fn label(i: &str) -> ParseResult<&str> {
    let loc = i
        .find(|c: char| !c.is_ascii_uppercase())
        .unwrap_or_else(|| i.len());
    if loc == 0 {
        return Err(format!("\"{}\" is not a valid label.", i));
    }
    let (label, i) = i.split_at(loc);
    Ok((i, label))
}

fn in_range(digits: &[u8], range: Range<u32>) -> Result<u32, String> {
    match std::str::from_utf8(digits).unwrap().parse() {
        Ok(i) if range.contains(&i) => Ok(i),
        _ => Err("Invalid register name".to_string()),
    }
}

fn register(i: &str) -> ParseResult<Register> {
    let non_alpha = i
        .find(|c: char| !c.is_alphanumeric())
        .unwrap_or_else(|| i.len());
    let (reg, i) = i.split_at(non_alpha);
    let reg = match reg.as_bytes() {
        b"zero" => 0,
        b"ra" => 1,
        b"sp" => 2,
        b"gp" => 3,
        b"tp" => 4,
        b"fp" => 8,
        [b'x', digits @ ..] => in_range(digits, 0..32)?,
        [b't', digits @ ..] => in_range(digits, 0..7).map(|i| match i {
            0..3 => i + 5,
            3..7 => i + 25,
            _ => unreachable!(),
        })?,
        [b's', digits @ ..] => in_range(digits, 0..12).map(|i| match i {
            0..2 => i + 8,
            2..12 => i + 16,
            _ => unreachable!(),
        })?,
        [b'a', digits @ ..] => in_range(digits, 0..8).map(|i| i + 10)?,
        [b'f', b't', digits @ ..] => in_range(digits, 0..12).map(|i| match i {
            0..8 => i,
            8..12 => i + 20,
            _ => unreachable!(),
        })?,
        [b'f', b's', digits @ ..] => in_range(digits, 0..12).map(|i| match i {
            0..2 => i + 8,
            2..12 => i + 16,
            _ => unreachable!(),
        })?,
        [b'f', b'a', digits @ ..] => in_range(digits, 0..8).map(|i| i + 10)?,
        [b'f', digits @ ..] => in_range(digits, 0..32)?,
        reg => {
            return Err(format!(
                "Invalid register name {}",
                std::str::from_utf8(reg).unwrap()
            ))
        }
    };
    Ok((i, Register(reg)))
}

fn immediate<T: TryFrom<i32, Error = String>>(i: &str) -> ParseResult<Immediate<'_, T>> {
    if let Ok((i, label)) = label(i) {
        return Ok((i, Immediate::Label(label)));
    }
    let (i, int) = signed_int(i)?;
    let imm = T::try_from(int).map(Immediate::Immediate)?;
    Ok((i, imm))
}

fn instruction(i: &str) -> ParseResult<Instruction> {
    let loc = i.find(is_space).unwrap_or_else(|| i.len());
    let (op, i) = i.split_at(loc);
    let i = i.trim_start_matches(is_space);
    Ok(match op {
        // pseudoinstructions
        "nop" => (
            i,
            Instruction::Raw {
                instruction: 0b000000000000_00000_000_00000_0010011,
            },
        ),
        "li" => {
            let (i, rd) = register(i)?;
            let i = i.trim_start_matches(arg_sep);
            let (i, imm) = immediate(i)?;
            (
                i,
                Instruction::IType {
                    opcode: 0b0010011,
                    funct3: 0b000,
                    destination: rd,
                    source1: Register(0b00000),
                    immediate: imm,
                },
            )
        }
        "mv" => translate_op(i, |rd, rs| Instruction::IType {
            opcode: 0b0010011,
            funct3: 0b000,
            destination: rd,
            source1: rs,
            immediate: Immediate::Immediate(I12(0)),
        })?,

        // RV32I
        "lui" => u_type(i, 0b0110111)?,
        "auipc" => u_type(i, 0b0010111)?,
        "jal" => j_type(i, 0b1101111)?,
        "jalr" => i_type(i, 0b1100111, 0b000)?,
        "beq" => b_type(i, 0b1100011, 0b000)?,
        "bne" => b_type(i, 0b1100011, 0b001)?,
        "blt" => b_type(i, 0b1100011, 0b100)?,
        "bge" => b_type(i, 0b1100011, 0b101)?,
        "bltu" => b_type(i, 0b1100011, 0b110)?,
        "bgeu" => b_type(i, 0b1100011, 0b111)?,
        "lb" => i_type(i, 0b0000011, 0b000)?,
        "lh" => i_type(i, 0b0000011, 0b001)?,
        "lw" => i_type(i, 0b0000011, 0b010)?,
        "lbu" => i_type(i, 0b0000011, 0b100)?,
        "lhu" => i_type(i, 0b0000011, 0b101)?,
        "sb" => s_type(i, 0b0100011, 0b000)?,
        "sh" => s_type(i, 0b0100011, 0b001)?,
        "sw" => s_type(i, 0b0100011, 0b010)?,
        "addi" => i_type(i, 0b0010011, 0b000)?,
        "slti" => i_type(i, 0b0010011, 0b010)?,
        "sltiu" => i_type(i, 0b0010011, 0b011)?,
        "xori" => i_type(i, 0b0010011, 0b100)?,
        "ori" => i_type(i, 0b0010011, 0b110)?,
        "andi" => i_type(i, 0b0010011, 0b111)?,
        "slli" => i_type(i, 0b0010011, 0b001)?,
        "srli" => i_type(i, 0b0010011, 0b101)?,
        "srai" => srai_op(i, 0b0010011)?,
        "add" => r_type(i, 0b0110011, 0b000, 0b0000000)?,
        "sub" => r_type(i, 0b0110011, 0b000, 0b0100000)?,
        "sll" => r_type(i, 0b0110011, 0b001, 0b0000000)?,
        "slt" => r_type(i, 0b0110011, 0b010, 0b0000000)?,
        "sltu" => r_type(i, 0b0110011, 0b011, 0b0000000)?,
        "xor" => r_type(i, 0b0110011, 0b100, 0b0000000)?,
        "srl" => r_type(i, 0b0110011, 0b101, 0b0000000)?,
        "sra" => r_type(i, 0b0110011, 0b101, 0b0100000)?,
        "or" => r_type(i, 0b0110011, 0b110, 0b0000000)?,
        "and" => r_type(i, 0b0110011, 0b111, 0b0000000)?,
        "fence" => i_type(i, 0b0001111, 0b000)?,
        "ecall" => (
            i,
            Instruction::Raw {
                instruction: 0b000000000000_00000_000_00000_1110011,
            },
        ),
        "ebreak" => (
            i,
            Instruction::Raw {
                instruction: 0b000000000001_00000_000_00000_1110011,
            },
        ),

        // RV64I
        "lwu" => i_type(i, 0b0000011, 0b110)?,
        "ld" => i_type(i, 0b0000011, 0b011)?,
        "sd" => s_type(i, 0b0100011, 0b011)?,
        "addiw" => i_type(i, 0b0011011, 0b000)?,
        "slliw" => i_type(i, 0b0011011, 0b001)?,
        "srliw" => i_type(i, 0b0011011, 0b101)?,
        "sraiw" => srai_op(i, 0b0011011)?,
        "addw" => r_type(i, 0b0111011, 0b000, 0b0000000)?,
        "subw" => r_type(i, 0b0111011, 0b000, 0b0100000)?,
        "sllw" => r_type(i, 0b0111011, 0b001, 0b0000000)?,
        "srlw" => r_type(i, 0b0111011, 0b101, 0b0000000)?,
        "sraw" => r_type(i, 0b0111011, 0b101, 0b0100000)?,
        op => return Err(format!("Invalid operation: {}", op)),
    })
}

fn r_type(i: &str, opcode: u32, funct3: u32, funct7: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs2) = register(i)?;
    Ok((
        i,
        Instruction::RType {
            opcode,
            funct3,
            funct7,
            destination: rd,
            source1: rs1,
            source2: rs2,
        },
    ))
}

// Float ops not implemented yet
#[allow(dead_code)]
fn r4_type(i: &str, opcode: u32, funct3: u32, funct2: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs2) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs3) = register(i)?;
    Ok((
        i,
        Instruction::R4Type {
            opcode,
            funct3,
            funct2,
            destination: rd,
            source1: rs1,
            source2: rs2,
            source3: rs3,
        },
    ))
}

fn i_type(i: &str, opcode: u32, funct3: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, imm) = immediate(i)?;
    Ok((
        i,
        Instruction::IType {
            opcode,
            funct3,
            destination: rd,
            source1: rs1,
            immediate: imm,
        },
    ))
}

fn s_type(i: &str, opcode: u32, funct3: u32) -> ParseResult<Instruction> {
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs2) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, imm) = immediate(i)?;
    Ok((
        i,
        Instruction::SType {
            opcode,
            funct3,
            source1: rs1,
            source2: rs2,
            immediate: imm,
        },
    ))
}

fn b_type(i: &str, opcode: u32, funct3: u32) -> ParseResult<Instruction> {
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs2) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, imm) = immediate(i)?;
    Ok((
        i,
        Instruction::BType {
            opcode,
            funct3,
            source1: rs1,
            source2: rs2,
            immediate: imm,
        },
    ))
}

fn u_type(i: &str, opcode: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, imm) = immediate(i)?;
    Ok((
        i,
        Instruction::UType {
            opcode,
            destination: rd,
            immediate: imm,
        },
    ))
}

fn j_type(i: &str, opcode: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, imm) = immediate(i)?;
    Ok((
        i,
        Instruction::JType {
            opcode,
            destination: rd,
            immediate: imm,
        },
    ))
}

fn translate_op<'a>(
    i: &'a str,
    f: fn(rd: Register, rs: Register) -> Instruction<'static>,
) -> ParseResult<'a, Instruction<'a>> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs) = register(i)?;
    Ok((i, f(rd, rs)))
}

fn srai_op(i: &str, opcode: u32) -> ParseResult<Instruction> {
    let (i, rd) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let (i, rs1) = register(i)?;
    let i = i.trim_start_matches(arg_sep);
    let loc = i
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or_else(|| i.len());
    let (digits, i) = i.split_at(loc);
    let imm = in_range(digits.as_bytes(), 0..64)? as i32;
    Ok((
        i,
        Instruction::IType {
            opcode,
            funct3: 0b101,
            destination: rd,
            source1: rs1,
            immediate: Immediate::Immediate(I12(imm | (1 << 30))),
        },
    ))
}

fn program(i: &str) -> ParseResult<Program<'_>> {
    let mut instructions = Vec::new();
    let mut labels = HashMap::new();
    for (row, line) in i.lines().enumerate() {
        let loc = instructions
            .len()
            .checked_shl(2)
            .ok_or_else(|| "Too many instructions.".to_string())?;
        let line = line.trim();
        let mut split = line.rsplitn(2, ':');
        let inst = split.next().unwrap();
        let l = split.next();
        if let Some(l) = l {
            let (i, l) = label(l)?;
            if i.len() > 1 {
                return Err(format!("Unexpected characters \"{}\" on line {}.", i, row));
            }
            if labels.insert(l, loc).is_some() {
                return Err(format!("Duplicate label {} on line {}.", l, row));
            }
        }
        if !inst.is_empty() {
            let (i, inst) = instruction(inst)?;
            if i.len() > 1 {
                return Err(format!("Unexpected characters \"{}\" on line {}.", i, row));
            }
            instructions.push(inst);
        }
    }
    Ok((
        i,
        Program {
            instructions,
            labels,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_simple() {
        let program = assemble(
            "
            LOOP:
            addi x1, x1, 1
            jal x0, LOOP
            ",
        )
        .unwrap();
        assert_eq!(
            program,
            vec![
                0b000000000001_00001_000_00001_0010011,
                0b1_1111111110_1_11111111_00000_1101111,
            ]
        )
    }
}
