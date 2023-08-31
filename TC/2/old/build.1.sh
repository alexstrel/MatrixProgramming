
#!/bin/bash
export QUDA_RESOURCE_PATH=.
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=1
# export MV2_SMP_USE_CMA=0
COARSEST_NC=64
ARGSTR="--dim 16 16 16 32 --prec single --prec-sloppy single --prec-precondition half --prec-null half --mg-smoother-halo-prec half --dslash-type asqtad --solve-type direct --compute-fat-long true --partition 15 --verbosity verbose --nsrc 2 "
ARGSTR=$ARGSTR"--mass 0.5 --tadpole-coeff 0.9 "
ARGSTR=$ARGSTR"--inv-multigrid true --mg-levels 3 --mg-coarse-solve-type 0 direct --mg-verbosity 0 verbose "
ARGSTR=$ARGSTR"--mg-setup-inv 1 cgnr --mg-setup-maxiter 1 10 --mg-setup-tol 1 1e-5 "
ARGSTR=$ARGSTR"--mg-block-size 0 2 2 2 2 --mg-nvec 0 24 --mg-coarse-solve-type 1 direct-pc --mg-smoother-solve-type 0 direct --mg-smoother 0 ca-gcr --mg-nu-pre 0 0 --mg-nu-post 0 8 --mg-smoother-tol 0 1e-10 --mg-coarse-solver-tol 1 5e-2 --mg-coarse-solver-maxiter 1 4   --mg-coarse-solver 1 gcr --mg-verbosity 1 verbose "
ARGSTR=$ARGSTR"--mg-block-size 1 2 2 2 2 --mg-nvec 1 ${COARSEST_NC} --mg-n-block-ortho 1 2 --mg-coarse-solve-type 2 direct-pc --mg-smoother-solve-type 1 direct-pc --mg-smoother 1 ca-gcr --mg-nu-pre 1 0 --mg-nu-post 1 2 --mg-smoother-tol 1 1e-10 --mg-coarse-solver-tol 2 0.25 --mg-coarse-solver-maxiter 2 16   --mg-coarse-solver 2 ca-gcr --mg-verbosity 2 verbose "
APP="tests/staggered_invert_test --mg-use-mma true $ARGSTR"
$APP


