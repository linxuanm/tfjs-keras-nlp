import * as tf from '@tensorflow/tfjs';

import { Activation } from '@tensorflow/tfjs-layers/dist/activations';
import { ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";
import { Initializer, InitializerIdentifier } from '@tensorflow/tfjs-layers/dist/initializers';
import { Constraint, ConstraintIdentifier } from '@tensorflow/tfjs-layers/dist/constraints';
import { Regularizer, RegularizerIdentifier } from '@tensorflow/tfjs-layers/dist/regularizers';


export type ActivationId
    = ActivationIdentifier
    | tf.serialization.ConfigDict
    | Activation;

export type ConstraintId
    = ConstraintIdentifier
    | tf.serialization.ConfigDict
    | Constraint;

export type InitializerId
    = InitializerIdentifier
    | Initializer
    | tf.serialization.ConfigDict;

export type RegularizerId
    = RegularizerIdentifier
    | tf.serialization.ConfigDict
    | Regularizer;
