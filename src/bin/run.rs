use std::{env, fs};
use riscv::{assembler::assemble, Computer};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let assembler = fs::read_to_string(&args[1]).unwrap();
    let program = assemble(&assembler).unwrap();
    let mut computer = Computer::new(args[2].parse().unwrap());
    computer.load_program(program);
    println!("{:?}", computer.run());
    print!("{}", computer.core.cycle_count);
}
