from deepxde.backend import tf
import os 

'''
This script is used to set tf parameters for the cluster
'''

tf.config.threading.set_intra_op_parallelism_threads(
    int(os.environ["TF_NUM_INTRAOP_THREADS"])
)
tf.config.threading.set_inter_op_parallelism_threads(
    int(os.environ["TF_NUM_INTEROP_THREADS"])
)
tf.config.set_soft_device_placement(True)