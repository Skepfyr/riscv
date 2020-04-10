//! The instruction format


/// A RISC-V General-purpose instruction
/// 
/// This structure intentionally doesn't have a one-to-one correspondance with
/// the RISC-V spec to make it more idiomatic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Instruction {
    /// Load from memory to an integer register 
    Load {
        /// The register to write the value to
        destination: Register,
        /// The register containing the base address into memory
        base: Register,
        /// The offset from the base address to load the falue from
        offset: i16,
        /// The width of the value to load
        width: Width,
        /// Whether to sign extend the value when writing it to destination
        sign_extend: bool,
    },
    /// Load from memory to an floating-point register 
    LoadFloat {
        /// The register to write the value to
        destination: FloatRegister,
        /// The register containing the base address into memory
        base: Register,
        /// The offset from the base address to load the falue from
        offset: i16,
        /// The width of the value to load
        width: Width,
    },
    MemoryFence {
        source: Register,
        destiation: Register,
        predecessor: MemoryOps,
        successor: MemoryOps,
        mode: FenceMode,
    },
    InstructionFence,
    Auipc {
        destination: Register,
        immediate: u32,
    },
    Store {
        width: Width,
        source1: Register,
        source2: Register,
        immediate: u32,
    },
    StoreFp {
        width: Width,
        source1: Register,
        source2: Register,
        immediate: u32,
    },
    AtomicMemoryOperation {
        atomic_type: AtomicOp,
        width: Width,
        source1: Register,
        source2: Register,
        destination: Register,
        acquire: bool,
        release: bool,
    },
    ArithmeticOp {
        operation: ArithmeticOp,
        source1: Register,
        source2: DataSource,
        destination: Register,
        width: Width,
    },
    MultiplicationOp {
        operation: MultiplicationOp,
        source1: Register,
        source2: Register,
        destiation: Register,
        width: Width,
    },
    Lui {
        destiation: Register,
        immediate: u32,
    },
    FusedMultiplyAdd {
        width: FloatWidth,
        source1: Register,
        source2: Register,
        source3: Register,
        destination: Register,
        rounding_mode: RoundingMode,
        negate_multiply: bool,
        negate_add: bool,
    },
    OpFloat {
        operation: FloatOp,
        source1: FloatRegister,
        source2: FloatRegister,
        destiation: FloatRegister,
        width: FloatWidth,
        rounding_mode: RoundingMode,
    },
    MoveFloatToInt {
        source: FloatRegister,
        destination: Register,
        float_width: FloatWidth,
        int_width: Option<Width>,
        rounding_mode: RoundingMode,
    },
    MoveIntToFloat {
        source: Register,
        destination: FloatRegister,
        int_width: Option<Width>,
        float_width: FloatWidth,
        rounding_mode: RoundingMode,
    },
    ConvertFloat {
        source: FloatRegister,
        destiation: FloatRegister,
        source_width: Width,
        destiation_width: Width,
        rounding_mode: RoundingMode,
    },
    Branch {
        test: Test,
        source1: Register,
        source2: Register,
        immediate: Register,
    },
    Jalr {
        source: Register,
        destiation: Register,
        immediate: u32,
    },
    Jal {
        destiation: Register,
        immediate: u32,
    },
    EnvironmentCall,
    EnvironmentBreakpoint,
    ControlStatusRegister {
        operation: CsrOperation,
        source: DataSource,
        destination: Register,
        csr: Csr,
    },
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    Byte = 0,
    HalfWord = 1,
    Word = 2,
    DoubleWord = 3,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatWidth {
    Single = 0,
    Double = 1,
    Half = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Register {
    register: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloatRegister {
    register: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryOps {
    device_input: bool,
    device_output: bool,
    memory_read: bool,
    memory_write: bool,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FenceMode {
    Normal = 0b0000,
    TotalStoreOrdering = 0b1000,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArithmeticOp {
    Add,
    Subtract,
    SetLessThan,
    SetLessThanUnsigned,
    Xor,
    Or,
    And,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MultiplicationOp {
    Multiply = 0,
    MultiplyUpper = 1,
    MultiplUpperSignedUnsigned = 2,
    MultiplyUpperUnsigned = 3,
    Divide = 4,
    DivideUnsigned = 5,
    Remainder = 6,
    RemainderUnsigned = 7,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicOp {
    Add = 0,
    Swap = 1,
    LoadReserved = 2,
    StoreConditional = 3,
    Xor = 4,
    Or = 8,
    And = 12,
    Min = 16,
    Max = 20,
    MinUnsigned = 24,
    MaxUnsigned = 28,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Sqrt,
    SignInject,
    SignInjectNot,
    SignInjectXor,
    Minimum,
    Maximum,
    Equal,
    LessThan,
    LessThanOrEqual,
    Class,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoundingMode {
    RoundToNearestEven = 0,
    RoundToZero = 1,
    RoundDown = 2,
    RoundUp = 3,
    RoundToNearestMax = 4,
    Dynamic = 7,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Test {
    Equal = 0,
    NotEqual = 1,
    LessThan = 4,
    GreaterThanOrEqual = 5,
    LessThanUnsigned = 6,
    GreaterThanOrEqualUnsigned = 7,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataSource {
    Immediate(u32),
    Register(Register),
}

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Csr {
    FloatFlags = 0x001,
    FloatRoundingMode = 0x002,
    FloatCsr = 0x003,
    Cycles = 0xC00,
    Time = 0xC01,
    InstructionsRetired = 0xC02,
    CyclesUpperWord = 0xC80,
    TimeUpperWord = 0xC81,
    InstructionsRetiredUpperWord = 0xC82,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CsrOperation {
    Write = 1,
    Set = 2,
    Clear = 3,
}
