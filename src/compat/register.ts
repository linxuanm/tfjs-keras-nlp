// Import this file to register all Tensorflow Serializable under compat.

import * as tfc from '@tensorflow/tfjs-core';

import { EinsumDense } from './einsum_dense';

tfc.serialization.registerClass(EinsumDense);
