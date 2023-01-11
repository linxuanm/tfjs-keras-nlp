
import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import { Activation, getActivation } from '@tensorflow/tfjs-layers/dist/activations';
import { Initializer, getInitializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { Constraint, getConstraint } from '@tensorflow/tfjs-layers/dist/constraints';
import { Regularizer, getRegularizer } from '@tensorflow/tfjs-layers/dist/regularizers';
import { getExactlyOneShape } from '@tensorflow/tfjs-layers/dist/utils/types_utils';

import {
    ActivationId, InitializerId, ConstraintId, RegularizerId
} from './keras_format';

export declare interface EinsumDenseLayerArgs {
    equation: string,
    output_shape: number | [number],
    activation?: ActivationId,
    bias_axes?: string,
    kernelInitializer?: InitializerId,
    biasInitializer?: InitializerId,
    kernelRegularizer?: RegularizerId,
    biasRegularizer?: RegularizerId,
    activityRegularizer?: RegularizerId,
    kernelConstraint?: ConstraintId,
    biasConstraint?: ConstraintId
};

export class EinsumDense extends tfl.layers.Layer {

    equation: string;
    partialOutputShape: tfl.Shape;
    biasAxes?: string;
    activation: Activation;
    kernelInitializer: Initializer;
    biasInitializer: Initializer;
    kernelRegularizer?: Regularizer;
    biasRegularizer?: Regularizer;
    kernelConstraint?: Constraint;
    biasConstraint?: Constraint;
    
    constructor(args: EinsumDenseLayerArgs) {
        super();
        
        this.equation = args.equation;
        if (typeof args.output_shape === "number")
            this.partialOutputShape = [args.output_shape];
        else
            this.partialOutputShape = args.output_shape;
        this.biasAxes = args.bias_axes;
        this.activation = getActivation(args.activation || "linear");
        this.kernelInitializer = getInitializer(args.kernelInitializer || "glorotUniform");
        this.biasInitializer = getInitializer(args.biasInitializer || "zeros");

        // ugly code, see https://github.com/tensorflow/tfjs/issues/7259
        this.kernelRegularizer = args.kernelRegularizer ? getRegularizer(args.kernelRegularizer) : undefined;
        this.biasRegularizer = args.biasRegularizer ? getRegularizer(args.biasRegularizer) : undefined;
        
        this.kernelConstraint = args.kernelConstraint ? getConstraint(args.kernelConstraint) : undefined;
        this.biasConstraint = args.biasConstraint ? getConstraint(args.biasConstraint) : undefined;
    }

    override build(inputShape: tfl.Shape | tfl.Shape[]): void {
        const oneShape = getExactlyOneShape(inputShape);
        const shapeData = analyzeEinsumString(
            this.equation, this.biasAxes, oneShape, this.partialOutputShape
        );
    }
}

interface SplitStringResult {
    weightShape: tfl.Shape,
    biasShape?: tfl.Shape,
    outputShape: tfl.Shape
}

function analyzeEinsumString(
    equation: string,
    biasAxes: string | undefined,
    inputShape: tfl.Shape,
    outputShape: tfl.Shape,
): SplitStringResult {
    const dotReplacedString = equation.replace(/\.\.\./g, '0');

    let splitString = dotReplacedString.match(/^([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)$/);
    if (splitString) {
        return analyzeSplitString(
            splitString, biasAxes, inputShape, outputShape
        );
    }

    splitString = dotReplacedString.match(/^0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)$/);
    if (splitString) {
        return analyzeSplitString(splitString, biasAxes, inputShape, outputShape, true);
    }

    splitString = dotReplacedString.match(/^([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0$/);
    if (splitString) {
        return analyzeSplitString(splitString, biasAxes, inputShape, outputShape);
    }

    throw new Error(
        `Invalid einsum equation '${equation}'. Equations must be in the form` +
        ` [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....`
    );
}

function analyzeSplitString(
    splitString: RegExpMatchArray,
    biasAxes: string | undefined,
    inputShape: tfl.Shape,
    outputShape: tfl.Shape,
    leftElided: boolean = false
): SplitStringResult {
    const inputSpec = splitString[1];
    const weightSpec = splitString[2];
    const outputSpec = splitString[3];
    const elided = inputShape.length - outputShape.length;

    outputShape.unshift(inputShape[0]);

    if (elided > 0 && leftElided) {
        for (let i = 1; i < elided; i++) {
            outputShape.splice(1, 0, inputShape[i]);
        }
    } else if (elided > 0 && !leftElided) {
        for (let i = inputShape.length - elided; i < inputShape.length; i++) {
            outputShape.push(inputShape[i]);
        }
    }

    const inputDimMap: { [key: string]: number; } = {};
    const outputDimMap: { [key: string]: number; } = {}
    if (leftElided) {
        [...inputSpec].forEach((dim, i) => inputDimMap[dim] = i + elided - inputShape.length);
        [...outputSpec].forEach((dim, i) => outputDimMap[dim] = i + elided);
    } else {
        [...inputSpec].forEach((dim, i) => inputDimMap[dim] = i);
        [...outputSpec].forEach((dim, i) => outputDimMap[dim] = i);
    }

    for (let dim of inputSpec) {
        const inputShapeAtDim = inputShape[inputDimMap[dim]];

        if (dim in outputDimMap) {
            const outputShapeAtDim = outputShape[outputDimMap[dim]];
            if (outputShapeAtDim != null && outputShapeAtDim !== inputShapeAtDim) {
                throw new Error(
                    `Input shape and output shape do not match at shared ` + 
                    `dimension '${dim}'. Input shape is ${inputShapeAtDim}, ` +
                    `and output shape ` +
                    `is ${outputShape[outputDimMap[dim]]}.`
                );
            }
        }
    }

    for (let dim of outputSpec) {
        if (!inputSpec.includes(dim) && !weightSpec.includes(dim)) {
            throw new Error(
                `Dimension '${dim}' was specified in the output ` +
                `'${outputSpec}' but has no corresponding dim in the input ` +
                `spec '${inputSpec}' or weight spec '${outputSpec}'`
            )
        }
    }

    const weightShape = [];
    for (let dim of weightSpec) {
        if (dim in inputDimMap) weightShape.push(inputShape[inputDimMap[dim]]);
        else if (dim in outputDimMap) weightShape.push(outputShape[outputDimMap[dim]]);
        else throw new Error(
            `Weight dimension '${dim}' did not have a match in either ` +
            `the input spec '${inputSpec}' or the output ` +
            `spec '${outputSpec}'. For this layer, the weight must ` +
            `be fully specified.`
        );
    }

    let biasShape: tfl.Shape | undefined;
    if (biasAxes != null) {
        const numLeftElided = leftElided ? elided : 0;
        const idxMap: { [key: string]: number } = {};
        [...outputSpec].forEach((char, i) => idxMap[char] = i + numLeftElided);

        for (let char of biasAxes) {
            if (!outputSpec.includes(char)) {
                throw new Error(
                    `Bias dimension '${char}' was requested, but is not part ` +
                    `of the output spec '${outputSpec}'`
                );
            }
        }

        const firstBiasLocation = Math.min(
            ...[...biasAxes].map(char => outputSpec.indexOf(char))
        );
        const biasOutputSpec = outputSpec.slice(firstBiasLocation);

        biasShape = [...biasOutputSpec].map(
            char => biasAxes.includes(char) ? idxMap[char] : 1
        );

        if (!leftElided) {
            biasShape = [...biasShape, ...new Array(elided).fill(1)];
        }
    }

    return { weightShape, outputShape, biasShape };
}
