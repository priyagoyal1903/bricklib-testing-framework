**Below are the commands that can be used to run the framework on NVIDIA summit:**

module load cuda/11.0.3 python/3.8.10 cmake/3.21.3 gcc/9.3.0
bsub -Is -W 02:00 -nnodes 1 -P CSC383 $SHELL
python python/main.py --backend cuda --config config.json
jsrun --nrs 1 --cpu_per_rs 7 --gpu_per_rs 1 --rs_per_host 1  --bind rs ./main
jsrun --nrs 1 --cpu_per_rs 7 --gpu_per_rs 1 --rs_per_host 1  --bind rs nvprof --metrics
flop_count_dp,dram_read_throughput,dram_write_throughput,achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput ./main

**Below commands can be used to run the framework on AMD crusher:**

module load cray-python rocm
salloc -A CSC383 -J hello -t 00:30:00 -p batch -N 1
python python/main.py --backend hip --config config.json
rocprof --timestamp on --stats -i rocprof-config.txt -o test.csv ./main

**To build the bricklib, below commands can be used.** (The installation path needs to be updated in framework's config.json file)

cd bricklib
rm build* -rf
mkdir build
cd build/
**1. For AMD crusher:**
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm-4.5.0/bin/amdclang++ -DUSE_HIP=ON
module load cuda/11.0.3 python/3.8.10 cmake/3.21.3 gcc/9.3.0
make single-hip
**2. For NVIDIA Summit:**
cmake .. -DCMAKE_CXX_COMPILER=g++
make single-cuda

**Below commands can be used to run the framework on Perlmutter:**

module load craype-accel-nvidia80 python/3.9-anaconda-2021.11 cmake cudatoolkit nvidia Nsight-Systems/2022.2.1
cd $SCRATCH
python python/main.py --backend cuda --config config.json
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 -c 1 --ntasks-per-node=1 --gpus-per-task=1 --threads-per-core=1 --gpu-bind=single:1 --account=m1411_g
dcgmi profile --pause
srun -n 1 ncu -f -o outputFile -s 1 -c 5 --set full ./main



