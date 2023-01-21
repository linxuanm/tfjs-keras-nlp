// ugly code, see https://github.com/tensorflow/tfjs/issues/7259
export function nullWrapper<Wrapped, Result>(
    func: (v: Wrapped) => Result, value?: Wrapped
): Result | null {
    return value === undefined ? null : func(value);
}

export function undefinedWrapper<Wrapped, Result>(
    func: (v: Wrapped) => Result, value?: Wrapped
): Result | undefined {
    return value === undefined ? undefined : func(value);
}

/* 
    Warning: brain damage ahead.

    So basically Tensorflow.js does not export its classes for
    components such as regularizers and constraints. Nor does
    it expose the necessary functions to (de)serialize one of
    the mentioned components. Therefore, the following is needed...

    Please raise an issue if there is a better method...
*/
import * as tf from "@tensorflow/tfjs";

import { ActivationId, InitializerId, ConstraintId, RegularizerId } from "./keras_format";

export type Kwargs = import("@tensorflow/tfjs-layers/dist/types").Kwargs;

export namespace TypeUtils {
    export function getExactlyOneShape(xs: tf.Shape | tf.Shape[]): tf.Shape {
        if (Array.isArray(xs)) {
            if (xs.length !== 1) throw `Expected Tensor length to be 1; got ${xs.length}`;

            return (<tf.Shape[]>xs)[0];
        }

        return xs;
    }

    export function getExactlyOneTensor(
        xs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]
    ): tf.Tensor<tf.Rank> {
        if (Array.isArray(xs)) {
            if (xs.length !== 1) throw `Expected Tensor length to be 1; got ${xs.length}`;

            return (<tf.Tensor<tf.Rank>[]>xs)[0];
        }

        return xs;
    }
}

export namespace Activations {
    export type Activation = import("@tensorflow/tfjs-layers/dist/activations").Activation;
    export type ActivationIdentifier = import("@tensorflow/tfjs-layers/dist/keras_format/activation_config").ActivationIdentifier;

    // TODO: rewrite this
    export function getActivation(identifier?: ActivationId): Activation {
        if (identifier == null) identifier = "linear";

        if (typeof identifier === "string") {
            return <Activation>tf.layers.activation({activation: identifier});
        } else if (identifier.constructor.name === "Activation") {
            return <Activation>identifier;
        }

        throw "Not yet implemented for ConfigDict!";
    }

    export function serializeActivation(activation: Activation) {
        return activation.getClassName();
    }
}

export namespace Constraints {
    export type Constraint = import("@tensorflow/tfjs-layers/dist/constraints").Constraint;
    export type ConstraintIdentifier = import("@tensorflow/tfjs-layers/dist/constraints").ConstraintIdentifier;

    // TODO: rewrite this
    export function getConstraint(identifier?: ConstraintId): Constraint {
        throw "Not yet implemented!";
    }

    export function serializeConstraint(constraint?: Constraint) {
        return serializeKerasObject(constraint);
    }
}

export namespace Regularizers {
    export type Regularizer = import("@tensorflow/tfjs-layers/dist/regularizers").Regularizer;
    export type RegularizerIdentifier = import("@tensorflow/tfjs-layers/dist/regularizers").RegularizerIdentifier;

    // TODO: rewrite this
    export function getRegularizer(identifier?: RegularizerId): Regularizer {
        throw "Not yet implemented!";
    }

    export function serializeRegularizer(regularizer?: Regularizer) {
        return serializeKerasObject(regularizer);
    }
}

export namespace Initializers {
    export type Initializer = import("@tensorflow/tfjs-layers/dist/initializers").Initializer;
    export type InitializerIdentifier = import("@tensorflow/tfjs-layers/dist/initializers").InitializerIdentifier;

    // TODO: rewrite this
    export function getInitializer(identifier?: InitializerId): Initializer {
        throw "Not yet implemented!";
    }

    export function serializeInitializer(initializer: Initializer) {
        return serializeKerasObject(initializer);
    }
}

export function serializeKerasObject(
    instance?: tf.serialization.Serializable
): tf.serialization.ConfigDictValue {
    if (instance == null) return null;

    return {
        className: instance.getClassName(),
        config: instance.getConfig()
    };
}
