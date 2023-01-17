import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { Initializer, getInitializer, serializeInitializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import { getExactlyOneShape, getExactlyOneTensor } from '@tensorflow/tfjs-layers/dist/utils/types_utils';

import { InitializerId } from '../../compat/keras_format';

const SEQUENCE_AXIS = -2;

export declare interface PositionalEmbeddingLayerArgs extends LayerArgs {
    sequenceLength: number,
    initializer?: InitializerId
};

export class PositionalEmbedding extends tfl.layers.Layer {

    sequenceLength: number;
    initializer: Initializer;
    positionalEmbeddings?: tfl.LayerVariable;

    constructor(args: PositionalEmbeddingLayerArgs) {
        super(args);
        this.sequenceLength = args.sequenceLength;
        this.initializer = getInitializer(args.initializer || "glorotUniform");
    }

    override getConfig(): tfc.serialization.ConfigDict {
        const config = {
            sequenceLength: this.sequenceLength,
            initializer: serializeInitializer(this.initializer)
        };

        const baseConfig = super.getConfig();

        return {...baseConfig, ...config};
    }

    override build(inputShape: tfl.Shape | tfl.Shape[]): void {
        const oneShape = getExactlyOneShape(inputShape);
        const featureSize = oneShape[oneShape.length - 1];

        this.positionalEmbeddings = this.addWeight(
            "embeddings",
            [this.sequenceLength, featureSize],
            this.dtype,
            this.initializer,
            undefined,
            true
        );

        super.build(inputShape);
    }

    override call(
        inputs: tfc.Tensor<tfc.Rank> | tfc.Tensor<tfc.Rank>[]
    ): tfc.Tensor<tfc.Rank> | tfc.Tensor<tfc.Rank>[] {
        return tfc.tidy(() => {
            const tensor = getExactlyOneTensor(inputs);

            // TF.js does not support RaggedTensor atm.
            return this.trimAndBroadcastPositionEmbeddings(tensor.shape);
        });
    }

    trimAndBroadcastPositionEmbeddings(shape: tfl.Shape): tfc.Tensor {
        const inputLength = shape[shape.length - SEQUENCE_AXIS]!;
        const positionEmbeddings = tfc.slice(
            this.positionalEmbeddings!.read(),
            [0, 0],
            [inputLength, -1]
        );

        return tfc.broadcastTo(positionEmbeddings, shape);
    }
}
