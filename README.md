
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                ____  _     ___ ____  _       _     
               | __ )| |   |_ _/ ___|| | __ _| |__  
               |  _ \| |    | |\___ \| |/ _` | '_ \ 
               | |_) | |___ | | ___) | | (_| | |_) |
               |____/|_____|___|____/|_|\__,_|_.__/ 

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

BLISlab: A Sandbox for Optimizing GEMM

.
├── README
├── results
│   ├── bl_dgemm_plot.m
│   ├── collect_result_step1.sh
│   ├── collect_result_step2.sh
│   ├── collect_result_step3.sh
│   ├── step1_result.m
│   ├── step2_result.m
│   └── step3_result.m
├── step1
│   ├── dgemm
│   │   ├── bl_dgemm_ref.c
│   │   ├── bl_dgemm_util.c
│   │   └── my_dgemm.c
│   ├── include
│   │   ├── bl_config.h
│   │   ├── bl_dgemm.h
│   │   └── bl_dgemm_ref.h
│   ├── lib
│   ├── makefile
│   ├── make.inc.files
│   │   ├── make.gnu.inc
│   │   ├── make.inc
│   │   └── make.intel.inc
│   ├── README
│   ├── sourceme.sh
│   └── test
│       ├── makefile
│       ├── run_bl_dgemm.sh
│       ├── tacc_run_bl_dgemm.sh
│       ├── test_bl_dgemm.c
│       └── test_bl_dgemm.x
├── step2
│   ├── dgemm
│   │   ├── bl_dgemm_ref.c
│   │   ├── bl_dgemm_util.c
│   │   └── my_dgemm.c
│   ├── include
│   │   ├── bl_config.h
│   │   ├── bl_dgemm.h
│   │   ├── bl_dgemm_kernel.h
│   │   └── bl_dgemm_ref.h
│   ├── kernels
│   │   ├── bl_dgemm_ukr.c
│   │   └── bli_dgemm_ukr.o
│   ├── lib
│   ├── makefile
│   ├── make.inc.files
│   │   ├── make.gnu.inc
│   │   ├── make.inc
│   │   └── make.intel.inc
│   ├── README
│   ├── sourceme.sh
│   └── test
│       ├── makefile
│       ├── run_bl_dgemm.sh
│       ├── tacc_run_bl_dgemm.sh
│       ├── test_bl_dgemm.c
│       └── test_bl_dgemm.x
└── step3
    ├── dgemm
    │   ├── bl_dgemm_ref.c
    │   ├── bl_dgemm_util.c
    │   └── my_dgemm.c
    ├── include
    │   ├── avx_types.h
    │   ├── bl_config.h
    │   ├── bl_dgemm.h
    │   ├── bl_dgemm_kernel.h
    │   └── bl_dgemm_ref.h
    ├── kernels
    │   ├── bl_dgemm_asm_12x4.c
    │   ├── bl_dgemm_asm_8x4.c
    │   ├── bl_dgemm_asm_8x6.c
    │   ├── bl_dgemm_int_8x4.c
    │   └── bl_dgemm_ukr.c
    ├── lib
    ├── makefile
    ├── make.inc.files
    │   ├── make.gnu.inc
    │   ├── make.inc
    │   └── make.intel.inc
    ├── README
    ├── sourceme.sh
    └── test
        ├── makefile
        ├── run_bl_dgemm.sh
        ├── tacc_run_bl_dgemm.sh
        ├── test_bl_dgemm.c
        └── test_bl_dgemm.x

How to compile and execute the code:
1. Change the options in sourceme.sh and set the environment variables.
$ source sourceme.sh
2. Compile the code, generate the library and test executables.
$ make
3. Execute the test driver.
$ cd test
$ ./run_bl_dgemm.sh

