import os
import re
import shutil
import subprocess
import shlex
import logging
import random
import string
from string import Template
import sys

import riscof.utils as utils
import riscof.constants as constants
from riscof.pluginTemplate import pluginTemplate
from riscv_isac.isac import isac

logger = logging.getLogger()

class spike(pluginTemplate):
    __model__ = "spike"
    __version__ = "1.1.0"

    def __init__(self, *args, **kwargs):
        sclass = super().__init__(*args, **kwargs)

        config = kwargs.get('config')
        if config is None:
            logger.error("Config node for spike missing.")
            raise SystemExit(1)
        self.ref_exe = os.path.join(config['PATH'] if 'PATH' in config else "","spike")
        self.num_jobs = str(config['jobs'] if 'jobs' in config else 1)
        self.plugin_path=os.path.abspath(config['plugin_path'])
        self.isa_spec = os.path.abspath(config['ispec']) if 'ispec' in config else ''
        self.platform_spec = os.path.abspath(config['pspec']) if 'ispec' in config else ''
        self.make = config['make'] if 'make' in config else 'make'
        logger.debug("spike plugin initialised using the following configuration.")
        for entry in config:
            logger.debug(entry+' : '+config[entry])
        return sclass

    def initialise(self, suite, work_dir, archtest_env):
        self.suite = suite
        if shutil.which(self.ref_exe) is None:
            logger.error('Please install Executable for DUTNAME to proceed further')
            raise SystemExit(1)
        self.work_dir = work_dir
        self.objdump_cmd = 'riscv{1}-elf-objdump -D {0} > {2};'
        self.compile_cmd = 'riscv{1}-elf-gcc -march={0} \
         -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles\
         -T '+self.plugin_path+'/env/link.ld\
         -I '+self.plugin_path+'/env/\
         -I ' + archtest_env

    def build(self, isa_yaml, platform_yaml):
        ispec = utils.load_yaml(isa_yaml)['hart0']
        self.xlen = ('64' if 64 in ispec['supported_xlen'] else '32')
        self.isa = 'rv' + self.xlen
        self.compile_cmd = self.compile_cmd+' -mabi='+('lp64 ' if 64 in ispec['supported_xlen'] else 'ilp32 ')
        if "I" in ispec["ISA"]:
            self.isa += 'i'
        if "M" in ispec["ISA"]:
            self.isa += 'm'
        if "C" in ispec["ISA"]:
            self.isa += 'c'
        if "F" in ispec["ISA"]:
            self.isa += 'f'
        if "D" in ispec["ISA"]:
            self.isa += 'd'
        objdump = "riscv{0}-elf-objdump".format(self.xlen)
        if shutil.which(objdump) is None:
            logger.error(objdump+": executable not found. Please check environment setup.")
            raise SystemExit(1)
        compiler = "riscv{0}-elf-gcc".format(self.xlen)
        if shutil.which(compiler) is None:
            logger.error(compiler+": executable not found. Please check environment setup.")
            raise SystemExit(1)
        if shutil.which(self.make) is None:
            logger.error(self.make+": executable not found. Please check environment setup.")
            raise SystemExit(1)

    def runTests(self, testList, cgf_file=None):
        if os.path.exists(self.work_dir+ "/Makefile." + self.name[:-1]):
            os.remove(self.work_dir+ "/Makefile." + self.name[:-1])
        make = utils.makeUtil(makefilePath=os.path.join(self.work_dir, "Makefile." + self.name[:-1]))
        make.makeCommand = self.make + ' -j' + self.num_jobs
        for file in testList:
            testentry = testList[file]
            test = testentry['test_path']
            test_dir = testentry['work_dir']
            test_name = test.rsplit('/',1)[1][:-2]

            elf = 'ref.elf'

            execute = "@cd "+testentry['work_dir']+";"

            cmd = self.compile_cmd.format(testentry['isa'].lower(), self.xlen) + ' ' + test + ' -o ' + elf
            compile_cmd = cmd + ' -D' + " -D".join(testentry['macros'])
            execute+=compile_cmd+";"

            execute += self.objdump_cmd.format(elf, self.xlen, 'ref.disass')
            sig_file = os.path.join(test_dir, self.name[:-1] + ".signature")

            execute += self.ref_exe + f' --isa={self.isa} +signature={sig_file} +signature-granularity=4 {elf} > {test_name}.log 2>&1;'

            cov_str = ' '
            for label in testentry['coverage_labels']:
               cov_str+=' -l '+label
            if cgf_file is not None:
               coverage_cmd = 'riscv_isac --verbose info coverage -d \
                       -t {0}.log --parser-name spike -o coverage.rpt  \
                       --sig-label begin_signature  end_signature \
                       --test-label rvtest_code_begin rvtest_code_end \
                       -e ref.elf -c {1} -x{2} {3};'.format(\
                       test_name, ' -c '.join(cgf_file), self.xlen, cov_str)
            else:
               coverage_cmd = ''
            execute+=coverage_cmd

            make.add_target(execute)
        make.execute_all(self.work_dir)
