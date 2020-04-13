use riscv::{Computer, assembler::assemble};

fn main() {
    let instructions = assemble(&std::fs::read_to_string("./res/primes.riscv").unwrap()).unwrap();
    let mut computer = Computer::new(1100);
    computer.load_program(instructions);
    computer.run();
}
