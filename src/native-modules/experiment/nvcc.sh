#!/bin/bash
echo "nvcc $@"
host_output=$1
shift
device_output=$1
shift

arch="-arch sm_61"

echo "host_output: $host_output"
echo "device_output: $device_output"

echo "nvcc $arch $@ $host_output"
nvcc $arch $@ -o $host_output
echo "nvcc $arch -Xcompiler -fPIC -dlink -o $device_output $host_output -lcudadevrt -lcudart"
nvcc $arch -Xcompiler -fPIC -dlink -o $device_output $host_output -lcudadevrt -lcudart
