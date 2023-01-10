
import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import { Activation, getActivation } from '@tensorflow/tfjs-layers/dist/activations';
import { Initializer, getInitializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { Constraint, getConstraint } from '@tensorflow/tfjs-layers/dist/constraints';
import { Regularizer, getRegularizer } from '@tensorflow/tfjs-layers/dist/regularizers';

import { ActivationId, InitializerId, ConstraintId, RegularizerId } from './keras_format';

export declare interface EinsumDenseLayerArgs {
    equation: string,
    output_shape: number | [number],
    activation?: ActivationId,
    bias_axes?: string,
    kernel_initializer?: InitializerId,
    bias_initializer?: InitializerId,
    kernel_regularizer?: RegularizerId,
    bias_regularizer?: RegularizerId,
    activity_regularizer?: RegularizerId,
    kernel_constraint?: ConstraintId,
    bias_constraint?: ConstraintId
};

export class EinsumDense extends tfl.layers.Layer {

    equation: string;
    partial_output_shape: [number];
    bias_axes?: string;
    activation: Activation;
    kernel_initializer: Initializer;
    bias_initializer: Initializer;
    kernel_regularizer?: Regularizer;
    bias_regularizer?: Regularizer;
    activity_regularizer?: Regularizer;
    kernel_constraint?: Constraint;
    bias_constraint?: Constraint;
    
    constructor(args: EinsumDenseLayerArgs) {
        super();
        
        this.equation = args.equation;
        if (typeof args.output_shape === "number")
            this.partial_output_shape = [args.output_shape];
        else
            this.partial_output_shape = args.output_shape;
        this.bias_axes = args.bias_axes;
        this.activation = getActivation(args.activation || "linear");
        this.kernel_initializer = getInitializer(args.kernel_initializer || "glorotUniform");
        this.bias_initializer = getInitializer(args.bias_initializer || "zeros");

        // ugly code, see https://github.com/tensorflow/tfjs/issues/7259
        this.kernel_regularizer = args.kernel_regularizer ? getRegularizer(args.kernel_regularizer) : undefined;
        this.bias_regularizer = args.bias_regularizer ? getRegularizer(args.bias_regularizer) : undefined;
        this.activity_regularizer = args.activity_regularizer ? getRegularizer(args.activity_regularizer) : undefined;
        
        this.kernel_constraint = args.kernel_constraint ? getConstraint(args.kernel_constraint) : undefined;
        this.bias_constraint = args.bias_constraint ? getConstraint(args.bias_constraint) : undefined;
    }
}
