import * as tfc from '@tensorflow/tfjs-core';

import { Activation } from '@tensorflow/tfjs-layers/dist/activations';
import { ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";
import { Initializer, InitializerIdentifier } from '@tensorflow/tfjs-layers/dist/initializers';
import { Constraint, ConstraintIdentifier } from '@tensorflow/tfjs-layers/dist/constraints';
import { Regularizer, RegularizerIdentifier } from '@tensorflow/tfjs-layers/dist/regularizers';


export type ActivationId
    = ActivationIdentifier
    | tfc.serialization.ConfigDict
    | Activation;

export type ConstraintId
    = ConstraintIdentifier
    | tfc.serialization.ConfigDict
    | Constraint;

export type InitializerId
    = InitializerIdentifier
    | Initializer
    | tfc.serialization.ConfigDict;

export type RegularizerId
    = RegularizerIdentifier
    | tfc.serialization.ConfigDict
    | Regularizer;
