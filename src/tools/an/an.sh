#!/bin/bash
echo "args: $@"
./ann $@
_exit_code=$?
echo "exitcode: $_exit_code"
