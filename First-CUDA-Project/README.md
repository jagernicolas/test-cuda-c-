##About
test applications/code for CUDA and documentation.

##Requirements
- NVidia CUDA 11.1 installed.
- Visual Studio 19 with C++ components installed.
- NVS installed into C:\Program Files\NVIDIA Corporation\NvToolsExt folder.

##Compile
open sln in Visual Studio and build in release mode. Please not that we aiming to run the apps on VMs which doesn't ship everything required to run apps in debug mode.

##Run
###local
copy `C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll` to `x64\Release` folder in your project folder.

###VM (AWS)
copy from your machine `C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll` in the same folder than `First-CUDA-Project.exe`
you can found in the doc folder `Create a new VM with CUDA and MS Visual Redistributable support.pdf` to setup a VM able to run the app. You can run the app with `--help` option to find options to setup.
