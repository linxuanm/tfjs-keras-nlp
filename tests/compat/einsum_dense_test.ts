import assert from "assert";

import * as tf from "@tensorflow/tfjs";

import { analyzeEinsumString } from "../../src/compat/einsum_dense";

interface EinsumTestData {
    testCaseName: string,
    equation: string,
    biasAxes?: string,
    inputShape: tf.Shape,
    outputShape: tf.Shape,
    expectedWeightShape: tf.Shape,
    expectedBiasShape?: tf.Shape,
    expectedOutputShape: tf.Shape
};

const data: EinsumTestData[] = [
    {
        testCaseName: "_1d_end_weight",
        equation: "ab,b->a",
        biasAxes: undefined,
        inputShape: [null, 32],
        outputShape: [],
        expectedWeightShape: [32],
        expectedBiasShape: undefined,
        expectedOutputShape: [null],
    },
    {
        testCaseName: "_2d_middle_weight",
        equation: "ab,bc->ac",
        biasAxes: undefined,
        inputShape: [null, 32],
        outputShape: [64],
        expectedWeightShape: [32, 64],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, 64]
    },
    {
        testCaseName: "_3d_bert",
        equation: "abc,cde->abde",
        biasAxes: undefined,
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, 1, 3, 4]
    },
    {
        testCaseName: "_3d_3_bias",
        equation: "abc,cde->abde",
        biasAxes: "e",
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [4],
        expectedOutputShape: [null, 1, 3, 4]
    },
    {
        testCaseName: "_3d_2_bias",
        equation: "abc,cde->abde",
        biasAxes: "d",
        inputShape: [null, 1, 2],
        outputShape: [1, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [3, 1],
        expectedOutputShape: [null, 1, 3, 4]
    },
    {
        testCaseName: "_3d_1_3_bias",
        equation: "abc,cde->abde",
        biasAxes: "be",
        inputShape: [null, 7, 2],
        outputShape: [7, 3, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: [7, 1, 4],
        expectedOutputShape: [null, 7, 3, 4]
    },
    {
        testCaseName: "_3d_bert_projection",
        equation: "BFNH,NHD->BFD",
        biasAxes: undefined,
        inputShape: [null, 1, 2, 3],
        outputShape: [1, 4],
        expectedWeightShape: [2, 3, 4],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, 1, 4]
    },
    {
        testCaseName: "_2d_bert",
        equation: "abc,cd->abd",
        biasAxes: undefined,
        inputShape: [null, 1, 2],
        outputShape: [1, 4],
        expectedWeightShape: [2, 4],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, 1, 4]
    },
    {
        testCaseName: "_embedding_1d",
        equation: "i,d->id",
        biasAxes: undefined,
        inputShape: [null],
        outputShape: [2],
        expectedWeightShape: [2],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, 2]
    },
    {
        testCaseName: "_xlnet_lm",
        equation: "ibd,nd->ibn",
        biasAxes: undefined,
        inputShape: [null, null, 1],
        outputShape: [null, 2],
        expectedWeightShape: [2, 1],
        expectedBiasShape: undefined,
        expectedOutputShape: [null, null, 2]
    }
];

describe('Einsum Dense', function() {
    describe('Test Weight Shapes', function() {
        data.forEach(v => {
            it(`Equation "${v.equation}" with bias axes "${v.biasAxes}"`, function() {
                const { weightShape, biasShape } = analyzeEinsumString(
                    v.equation, v.biasAxes, v.inputShape, v.outputShape
                );

                assert.deepEqual(weightShape, v.expectedWeightShape);
                assert.deepEqual(biasShape, v.expectedBiasShape);
            });
        });
    });
});
