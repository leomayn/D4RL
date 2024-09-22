# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['MUJOCO_GL'] = 'egl'
CUDA_VISIBLE_DEVICES=0
PYOPENGL_PLATFORM=egl
MUJOCO_GL=egl
# repeat python visualiation.py for 10 times with different rng seeds
for i in {1..10}
do
    echo "RNG=$i"
    python algos/visualization.py RNG=$i
done