import {NNNetwork, NNNetworkObj} from "../structures/Network";

export interface ManipulationData {
    bias: number;
    weights: number[];
}

export class ManipulationMatrix {
    changes: ManipulationData[][];

    constructor(network: NNNetworkObj) {
        this.changes = [];
        for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
            const nodeCount = network.layers[layerIndex].nodes.length;
            this.changes[layerIndex] = [];
            for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
                this.changes[layerIndex].push({bias: 0, weights: []});
            }
        }
    }

    addBias(value: number, layerIndex: number, nodeIndex: number) {
        this.changes[layerIndex][nodeIndex].bias += value;
    }

    addWeight(value: number, layerIndex: number, nodeIndex: number, weightIndex: number) {
        if (!this.changes[layerIndex][nodeIndex].weights[weightIndex]) {
            this.changes[layerIndex][nodeIndex].weights[weightIndex] = value;
        } else {
            this.changes[layerIndex][nodeIndex].weights[weightIndex] += value;
        }
    }

    export(): ManipulationData[][] {
        return this.changes;
    }

    apply(network: NNNetwork, biasLearningRate: number, weightLearningRate: number) {
        const layerCount = network.layers.length;
        for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {

            const nodeCount = network.layers[layerIndex].nodes.length;

            for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {

                const weightCount = network.layers[layerIndex].nodes[nodeIndex].weights.length;

                for (let weightIndex = 0; weightIndex < weightCount; weightIndex++) {

                    const weightDiff = this.changes[layerIndex][nodeIndex].weights[weightIndex] * weightLearningRate;

                    network.layers[layerIndex].nodes[nodeIndex].weights[weightIndex] -= weightDiff;
                }

                const biasDiff = this.changes[layerIndex][nodeIndex].bias * biasLearningRate;

                network.layers[layerIndex].nodes[nodeIndex].bias -= biasDiff;
            }
        }
    }
}