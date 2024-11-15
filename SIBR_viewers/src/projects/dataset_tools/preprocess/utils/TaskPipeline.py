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
import re
import shutil
from importlib import import_module
from utils.convert import updateStringFromDict
from utils.commands import runCommand

class TaskPipeline:
    def __init__(self, args, steps, programs):
        self.args = args
        self.steps = steps
        self.programs = programs

    def isExpressionValid(self, expression):
        if not re.match(r"^((?:not|and|or|is|in|\$\{\w+\})+\s*)+$", expression):
            print("Invalid expression '%s'." % expression)

        return eval(updateStringFromDict(expression, self.args))

    def runProcessSteps(self):
        for step in self.steps:
            if "if" in step and not self.isExpressionValid(step["if"]):
                print("Nothing to do on step %s. Skipping." % (step["name"]))
                continue

            print("Running step %s..." % step["name"])
            if "app" in step:
                if self.args["dry_run"]:
                    print('command : %s %s' % (self.programs[step["app"]]["path"], ' '.join([updateStringFromDict(command_arg, self.args) for command_arg in step["command_args"]])))

                    success = True
                else:
                    completedProcess = runCommand(self.programs[step["app"]]["path"], [updateStringFromDict(command_arg, self.args) for command_arg in step["command_args"]])

                    success = completedProcess.returncode == 0

            elif "function" in step:
                if '.' in step["function"]:
                    currentModuleName, currentFunctionName = step["function"].rsplit('.', 1)
                    currentFunction = getattr(import_module(currentModuleName), currentFunctionName)
                else:
                    print("Missing module name for function %s. Aborting." % (step["function"]))
                    sys.exit(1)

                if self.args["dry_run"]:
                    print('function : %s(%s)' % (step["function"], ', '.join([ "%s=%s" % (key, ([updateStringFromDict(item, self.args) for item in val]
                                                                                                 if type(val) is list else
                                                                                                 updateStringFromDict(val, self.args)))
                                                                                                    for key, val in step["function_args"].items()])))
                else:
                    currentFunction(**{ key: ([updateStringFromDict(item, self.args) for item in val]
                                                if type(val) is list else
                                                updateStringFromDict(val, self.args))
                                                    for key, val in step["function_args"].items() })

                success = True
            else:
                print("Nothing to do on step %s. Skipping." % (step["name"]))
                continue

            if success:
                print("Step %s successful." % (step["name"]))
            else:
                sys.stdout.flush()
                sys.stderr.flush()
                print("Error on step %s. Aborting." % (step["name"]))
                sys.exit(1)