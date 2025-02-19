use riscv::{assembler::assemble, Computer, Memory};
use std::{env, fs};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let assembler = fs::read_to_string(&args[1]).unwrap();
    let program = assemble(&assembler).unwrap();
    let mut computer = Computer::new(args[2].parse().unwrap());
    computer.memory.main[..program.len()].copy_from_slice(&program);
    let boot_code = assemble(&format!(
        "
        lui t1, 0x{mem_start:x}
        jalr x0, t1, 0
        ",
        mem_start = Memory::RAM_OFFSET >> 12
    ))
    .unwrap();
    computer.memory.rom[riscv::RESET_VECTOR as usize..][..boot_code.len()]
        .copy_from_slice(&boot_code);
    println!("{:?}", computer.run());
}
