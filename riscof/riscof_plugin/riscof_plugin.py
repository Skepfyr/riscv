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

logger = logging.getLogger()


class plugin(pluginTemplate):
    __model__ = "risc-v"
    __version__ = "dev"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get("config") or dict()

        self.dut_exe = os.path.join(
            os.path.abspath(config["PATH"]) if "PATH" in config else "", "test"
        )

        # Number of parallel jobs that can be spawned off by RISCOF
        # for various actions performed in later functions, specifically to run the tests in
        # parallel on the DUT executable. Can also be used in the build function if required.
        self.num_jobs = str(config["jobs"] if "jobs" in config else 1)
        self.plugin_path = os.path.abspath(config["plugin_path"])
        self.isa_spec = os.path.join(self.plugin_path, "./isa.yaml")
        self.platform_spec = os.path.join(self.plugin_path, "./platform.yaml")

        # We capture if the user would like the run the tests on the target or
        # not. If you are interested in just compiling the tests and not running
        # them on the target, then following variable should be set to False
        if "target_run" in config and config["target_run"] == "0":
            self.target_run = False
        else:
            self.target_run = True

    def initialise(self, suite, work_dir, archtest_env):

        # capture the architectural test-suite directory.
        self.suite = suite

        # capture the working directory where all the execution and meta
        # files/states should be dumped
        self.work_dir = work_dir

        # capture the path to the architectural test environment.
        self.archtest_env = archtest_env

    def build(self, isa_yaml, platform_yaml):
        pass

    def runTests(self, testList):
        # Delete Makefile if it already exists.
        if os.path.exists(self.work_dir + "/Makefile." + self.name[:-1]):
            os.remove(self.work_dir + "/Makefile." + self.name[:-1])
        # create an instance the makeUtil class that we will use to create targets.
        make = utils.makeUtil(
            makefilePath=os.path.join(self.work_dir, "Makefile." + self.name[:-1])
        )

        # set the make command that will be used. The num_jobs parameter was set in the __init__
        # function earlier
        make.makeCommand = "make -k -j" + self.num_jobs

        # we will iterate over each entry in the testList. Each entry node will be refered to by the
        # variable testname.
        for file in testList:

            # for each testname we get all its fields (as described by the testList format)
            testentry = testList[file]

            # we capture the path to the assembly file of this test
            test = testentry["test_path"]

            # capture the directory where the artifacts of this test will be dumped/created. RISCOF is
            # going to look into this directory for the signature files
            test_dir = testentry["work_dir"]
            test_name = test.rsplit('/',1)[1][:-2]

            # name of the elf file after compilation of the test
            elf = "my.elf"

            # name of the signature file as per requirement of RISCOF. RISCOF expects the signature to
            # be named as DUT-<dut-name>.signature. The below variable creates an absolute path of
            # signature file.
            sig_file = os.path.join(test_dir, self.name[:-1] + ".signature")

            # for each test there are specific compile macros that need to be enabled. The macros in
            # the testList node only contain the macros/values. For the gcc toolchain we need to
            # prefix with "-D".
            compile_macros = "".join(" -D" + macro for macro in testentry["macros"])

            cmd = (
                f"riscv64-elf-gcc -march={testentry['isa'].lower()} -mabi=lp64 "
                f"-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -g "
                f"-T {self.plugin_path}/env/link.ld "
                f"-I {self.plugin_path}/env/ "
                f"-I {self.archtest_env} "
                f"{test} -o {elf} {compile_macros} "
            )

            # if the user wants to disable running the tests and only compile the tests, then
            # the "else" clause is executed below assigning the sim command to simple no action
            # echo statement.
            if self.target_run:
                # set up the simulation command. Template is for spike. Please change.
                simcmd = f"{self.dut_exe} {elf} {sig_file} > {test_name}.log 2>&1"
            else:
                simcmd = 'echo "NO RUN"'

            # concatenate all commands that need to be executed within a make-target.
            execute = f"@cd {testentry["work_dir"]}; {cmd}; {simcmd};"

            # create a target. The makeutil will create a target with the name "TARGET<num>" where num
            # starts from 0 and increments automatically for each new target that is added
            make.add_target(execute)

        # if you would like to exit the framework once the makefile generation is complete uncomment the
        # following line. Note this will prevent any signature checking or report generation.
        # raise SystemExit

        # once the make-targets are done and the makefile has been created, run all the targets in
        # parallel using the make command set above.
        make.execute_all(self.work_dir)

        # if target runs are not required then we simply exit as this point after running all
        # the makefile targets.
        if not self.target_run:
            raise SystemExit(0)
