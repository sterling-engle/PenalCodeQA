:: Run NVIDIA Nsight Compute CLI (ncu):
:: -o penalcode	  output to penalcode.ncu-rep
:: [-k kernal_name]
:: --set full	collect the full set of sections
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2021.1.0\target\windows-desktop-win7-x64\ncu.exe" -o penalcode --set full C:\ProgramData\Anaconda3\python.exe penalcode.py --skip=299