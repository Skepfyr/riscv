use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write as _},
};

use elf::{endian::LittleEndian, ElfBytes};
use riscv::{assembler::assemble, Computer, Memory};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let file = fs::read(&args[1]).unwrap();
    let elf = ElfBytes::<LittleEndian>::minimal_parse(&file).unwrap();
    let mut computer = Computer::new(0x1_000_000);
    for segment in elf.segments().unwrap().iter() {
        if segment.p_type == elf::abi::PT_LOAD {
            computer.memory.main.get_mut().unwrap()
                [segment.p_paddr as usize - Memory::RAM_OFFSET..][..segment.p_memsz as usize]
                .copy_from_slice(elf.segment_data(&segment).unwrap());
        }
    }
    let (symbols, strings) = elf.symbol_table().unwrap().unwrap();
    let entrypoint = symbols
        .iter()
        .find(|sym| {
            sym.st_name > 0 && strings.get(sym.st_name as usize).unwrap() == "rvtest_entrypoint"
        })
        .unwrap()
        .st_value;
    let signature_start = symbols
        .iter()
        .find(|sym| {
            sym.st_name > 0 && strings.get(sym.st_name as usize).unwrap() == "begin_signature"
        })
        .unwrap()
        .st_value;
    let signature_end = symbols
        .iter()
        .find(|sym| {
            sym.st_name > 0 && strings.get(sym.st_name as usize).unwrap() == "end_signature"
        })
        .unwrap()
        .st_value;

    let entrypoint_high_high = entrypoint >> (32 + 12);
    let entrypoint_high_low = (entrypoint >> 32) & 0xfff;
    let entrypoint_low_high = entrypoint >> 12;
    let entrypoint_low_low = entrypoint & 0xfff;
    let instructions = assemble(&format!(
        "
        mv t1, x0
        lui t1, 0x{entrypoint_high_high:x}
        addi t1, t1, 0x{entrypoint_high_low:x}
        slli t1, t1, 32
        lui t1, 0x{entrypoint_low_high:x}
        jalr x0, t1, 0x{entrypoint_low_low:x}
        "
    ))
    .unwrap();
    computer.memory.rom[riscv::RESET_VECTOR as usize..][..instructions.len()]
        .copy_from_slice(&instructions);
    computer.run().unwrap();

    let mut sig_file = BufWriter::new(
        File::options()
            .create_new(true)
            .write(true)
            .open(&args[2])
            .unwrap(),
    );
    let sig_start = signature_start as usize - 0x40000000;
    let sig_end = signature_end as usize - 0x40000000;
    for signature in computer.memory.main.get_mut().unwrap()[sig_start..sig_end].chunks_exact(4) {
        writeln!(
            sig_file,
            "{:08x}",
            u32::from_le_bytes(signature.try_into().unwrap())
        )
        .unwrap();
    }
    sig_file.flush().unwrap();
}
