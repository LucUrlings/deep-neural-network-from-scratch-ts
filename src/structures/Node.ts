import {Sigmoid} from "../math/activation-functions";

export interface NNNodeObj {
    weights: number[];
    bias: number;
}

export class NNNode implements NNNodeObj {
    weights: number[];
    bias: number;

    zValue: number;
    activity: number;

    constructor() {
        this.weights = [];
        this.bias = 0;
        this.activity = 0;
        this.zValue = 0;
    }

    init(numberOfWeights: number, weightsRange: number, biasRange: number): void {
        for (let i = 0; i < numberOfWeights; i++) {
            this.weights.push((Math.random() - 0.5) * weightsRange);
        }

        this.bias = (Math.random() - 0.5) * biasRange;
    }

    load(data: NNNodeObj): void {
        this.weights = data.weights;
        this.bias = data.bias;
    }

    export(): NNNodeObj {
        return {
            weights: this.weights,
            bias: this.bias
        }
    }

    calculateValue(inputActivations: number[]): number {
        if (inputActivations.length !== this.weights.length) {
            throw new Error(`Length of inputs (${inputActivations.length}) does not match length of weights (${this.weights.length})`);
        }

        let sum = 0;
        for (let i = 0; i < inputActivations.length; i++) {
            sum += inputActivations[i] * this.weights[i];
        }
        sum += this.bias;
        this.zValue = sum;
        const result = Sigmoid(sum);
        this.activity = result;
        return result;
    }
}