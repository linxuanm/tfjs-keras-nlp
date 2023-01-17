// Import this file to register all Tensorflow Serializable under compat.

import * as tf from '@tensorflow/tfjs';

import { EinsumDense } from './einsum_dense';

tf.serialization.registerClass(EinsumDense);
