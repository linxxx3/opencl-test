# OpenCL Test

Simple OpenCL tests.

## Requirements

* Platform: `x86_64/aarch64 + NV GPU`

* OS: Ubuntu 18.04

* NVidia driver: `NVIDIA-Linux-aarch64-450.80.02.run`

* Packages:

```
sudo apt install git gcc mesa-opencl-icd ocl-icd-opencl-dev
```

## Compile

* on `x86_64`

```
gcc hello.c -o hello -O2 /usr/lib/x86_64-linux-gnu/libOpenCL.so.1
```

* on aarch64

```
gcc hello.c -o hello -O2 /usr/lib/aarch64-linux-gnu/libOpenCL.so.1
```

## Run

```
./hello
```
