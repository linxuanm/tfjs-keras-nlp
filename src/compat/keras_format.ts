import * as tf from '@tensorflow/tfjs';

import { Activations, Constraints, Initializers, Regularizers } from "./tfjs_fix";

export type ActivationId
    = Activations.ActivationIdentifier
    | tf.serialization.ConfigDict
    | Activations.Activation;

export type ConstraintId
    = Constraints.ConstraintIdentifier
    | tf.serialization.ConfigDict
    | Constraints.Constraint;

export type InitializerId
    = Initializers.InitializerIdentifier
    | Initializers.Initializer
    | tf.serialization.ConfigDict;

export type RegularizerId
    = Regularizers.RegularizerIdentifier
    | tf.serialization.ConfigDict
    | Regularizers.Regularizer;
