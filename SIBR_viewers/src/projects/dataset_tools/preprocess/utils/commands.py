# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

import subprocess
import os, sys
from utils.paths import getBinariesPath, getColmapPath

def getProcess(programName, binaryPath = getBinariesPath()):
    suffixes = [ '', '_msr', '_rwdi', '_d']

    for suffix in suffixes:
        binary = os.path.join(binaryPath, programName + suffix + ".exe")

        if os.path.isfile(binary):
            print("Program '%s' found in '%s'." % (programName, binary))
            return binary

def runCommand(binary, command_args):
    print("Running process '%s'" % (' '.join([binary, *command_args])))
    sys.stdout.flush()
    completedProcess = subprocess.run([binary, *command_args])

    if completedProcess.returncode == 0:
        print("Process %s completed." % binary)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        print("Process %s failed with code %d." % (binary, completedProcess.returncode))

    return completedProcess

def getColmap(colmapPath = getColmapPath()):
    colmapBinary = os.path.join(colmapPath, "COLMAP.bat")

    if os.path.isfile(colmapBinary):
        print("Program '%s' found in '%s'." % (colmapBinary, colmapPath))
        return colmapBinary
    else:
        print("Program '%s' not found in '%s'. Aborting." % (colmapBinary, colmapPath))
        return None