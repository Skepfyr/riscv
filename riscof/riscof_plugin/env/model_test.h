#ifndef _COMPLIANCE_MODEL_H
#define _COMPLIANCE_MODEL_H

//RV_COMPLIANCE_DATA_BEGIN
#define RVMODEL_DATA_BEGIN \
  .align 4; \
  .global begin_signature; begin_signature:

//RV_COMPLIANCE_DATA_END
#define RVMODEL_DATA_END \
  .align 4; \
  .global end_signature; end_signature:

//RV_COMPLIANCE_HALT
#define RVMODEL_HALT \
  li x1, 0x2000;     \
  li x2, 0x4000;       \
  halt:              \
    sw x1, 0(x2);\
    j halt;

#define RVMODEL_BOOT

#define RVMODEL_SET_MSW_INT

#define RVMODEL_CLEAR_MSW_INT

#define RVMODEL_CLEAR_MTIMER_INT

#define RVMODEL_CLEAR_MEXT_INT

#define RVMODEL_MTVEC_ALIGN 2

//RVTEST_IO_INIT
#define RVMODEL_IO_INIT
//RVTEST_IO_WRITE_STR
#define RVMODEL_IO_WRITE_STR(_R, _STR)
//RVTEST_IO_CHECK
#define RVMODEL_IO_CHECK()
//RVTEST_IO_ASSERT_GPR_EQ
#define RVMODEL_IO_ASSERT_GPR_EQ(_S, _R, _I)
//RVTEST_IO_ASSERT_SFPR_EQ
#define RVMODEL_IO_ASSERT_SFPR_EQ(_F, _R, _I)
//RVTEST_IO_ASSERT_DFPR_EQ
#define RVMODEL_IO_ASSERT_DFPR_EQ(_D, _R, _I)

#endif // _COMPLIANCE_MODEL_H
