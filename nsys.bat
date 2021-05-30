:: Run NVIDIA Nsight Systems profiling:
:: -f true
:: -o penalcode1			sets .qdstrm filename
:: --sample none			do not collect CPU samples
:: --trace cuda,cublas			only trace cuda and cublas
:: --capture-range cudaProfilerApi	start only when application calls profiler.start() 
:: --capture-range-end stop		stop capturing when application calls profiler.stop()
:: --export sqlite			creates additional output files
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2021.2.1\target-windows-x64\nsys.exe" profile -f true -o penalcode1 --sample none --trace cuda,cublas --capture-range cudaProfilerApi --capture-range-end stop --export sqlite C:\ProgramData\Anaconda3\python.exe penalcode.py --skip=299 --pyprof