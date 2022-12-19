from generate_constants import gen_consts_file
from kernel_config_application import KernelConfigApplier
from argparse import ArgumentParser
import os
import json
import subprocess

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--backend", type=str, required=True, choices=['hip', 'cuda'])

    args = parser.parse_args()
    cwd = os.getcwd()
    config_path = os.path.join(cwd, args.config)
    with open(config_path, 'r') as f:
        decoded = json.load(f)
    
    gen_consts_file(args.backend, args.config)
    kernels = decoded["kernels"]
    kernel_objects = map(
        lambda k: KernelConfigApplier(k, kernels[k]["versions"], (kernels[k]["sizes"] if "sizes" in kernels[k] else None), args.backend),
        kernels.keys()
    )

    vecscatter_path = os.path.join(decoded["bricklib-path"], "codegen", "vecscatter")
    brick_include_path = os.path.join(decoded["bricklib-path"], "include")
    to_call = []
    for k in kernel_objects:
        e = (decoded[args.backend]["codegen-flags"] if "codegen-flags" in decoded[args.backend] else [])
        if "--" not in e:
            e.append("--")
        e.append("-I" + brick_include_path)
        k.generate_intermediate_code().run_codegen_vecscatter(vecscatter_path, python="python3", extras=e)
        to_call.extend(k.wrap_functions())
    
    os.makedirs(os.path.join(cwd, "out"), exist_ok=True)
    compile_files=[]
    with open(os.path.join(cwd, "out", "main.cu"), "w") as f:
        f.write('#include "./gen/consts.h"\n')
        f.write(f'#include <brick-{args.backend}.h>\n')
        f.write("""#include <iostream>
#include \"bricksetup.h\"
#include \"multiarray.h\"
#include \"brickcompare.h\"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <string.h>
#include \"brick.h\"
#include \"vecscatter.h\"
""")
        for kernel_name in kernels.keys():
            f.write(f'#include "./kernels/{kernel_name}/gen/{kernel_name}.h"\n')
            compile_files.append(f"./kernels/{kernel_name}/gen/{kernel_name}.cu")
            
        f.write("typedef void (*kernel)();\n")
        f.write("int main(void) {\n")
        f.write("\t kernel kernels[] = {")
        f.write(",".join(to_call))
        f.write("};\n")

        
        f.write(f"\tfor (int i = 0; i < {len(to_call)}; i++) " + "{\n")
        f.write("\t\tkernels[i]();\n")
        f.write("\t}\n")

        f.write("}\n")

    extra_flags = (decoded[args.backend]["compiler-flags"] if "compiler-flags" in decoded[args.backend] else [])
    command = [decoded[args.backend]["compiler"], "out/main.cu", *compile_files, "-I", f'{decoded["bricklib-path"]}/include', "-L", f'{decoded["bricklib-path"]}/build/src', '-l', 'brickhelper', '-o', (decoded["output"] if "output" in decoded else "main"), *extra_flags]    
    print(" ".join(command))
    subprocess.run(command)
