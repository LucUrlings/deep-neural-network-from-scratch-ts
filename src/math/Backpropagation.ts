import {NNNetworkObj} from "../structures/Network";
import {Memo} from "./Memo";
import {Sigmoid} from "./activation-functions";
import {ManipulationMatrix} from "./Manipulation";

export class Backpropagation {
    network: NNNetworkObj;
    expected: number[];

    constructor(network: NNNetworkObj, expected: number[]) {
        this.network = network;
        this.expected = expected;
    }

    get costArray(): number[] {
        const costArray = [];
        const output = this.network.layers[this.network.layers.length - 1].nodeActivity;
        if (output.length !== this.expected.length) {
            throw new Error(`Ouput length (${output.length}) and expected length (${this.expected.length}) don't match!`);
        }
        for (let i = 0; i < output.length; i++) {
            costArray.push(Math.pow(output[i] - this.expected[i], 2) / 2);
        }
        return costArray
    }

    get totalCost(): number {
        const costArray = this.costArray;
        return costArray.reduce((a, b) => a + b);
    }

    deltaPrevActivityOnZ(layerIndex: number, currentNodeIndex: number, prevNodeIndex: number) {
        return this.network.layers[layerIndex].nodes[currentNodeIndex].weights[prevNodeIndex];
    }

    deltaBiasOnZ() {
        return 1;
    }

    deltaWeightOnZ(layerIndex: number, weightIndex: number): number {
        if (layerIndex !== 0) {
            return this.network.layers[layerIndex - 1].nodes[weightIndex].activity;
        } else {
            return this.network.input && this.network.input[weightIndex] || 0;
        }
    }

    deltaZOnActivity(layerIndex: number, nodeIndex: number) {
        const zValue = this.network.layers[layerIndex].nodes[nodeIndex].zValue;
        const sig = Sigmoid(zValue);
        return (sig * (1 - sig));
    }

    deltaActivityOnCost(layerIndex: number, nodeIndex: number, expectedIndex: number) {
        return (this.network.layers[layerIndex].nodes[nodeIndex].activity - this.expected[expectedIndex]);
    }

    execute(manipulationMatrix: ManipulationMatrix) {
        const memo = new Memo();

        memo.init(this.network.layers.length);

        const layerCount = this.network.layers.length - 1;

        for (let layerIndex = layerCount; layerIndex >= 0; layerIndex--) {
            const nodeCount = this.network.layers[layerIndex].nodes.length;

            for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
                const expectedIndex = nodeIndex;

                if (layerIndex === layerCount) {
                    const activityOnCost = this.deltaActivityOnCost(layerIndex, nodeIndex, expectedIndex);
                    memo.add(activityOnCost, layerIndex, nodeIndex);
                }

                const biasOnZ = this.deltaBiasOnZ();
                const zOnActivity = this.deltaZOnActivity(layerIndex, nodeIndex);

                const activityOnCost = memo.get(layerIndex, nodeIndex);

                const biasOnCost = biasOnZ * zOnActivity * activityOnCost;

                manipulationMatrix.addBias(biasOnCost, layerIndex, nodeIndex);

                console.log(`Effect of increasing bias of layer[${layerIndex}].nodes[${nodeIndex}] on error: ${biasOnCost}`);

                const weightCount = this.network.layers[layerIndex].nodes[nodeIndex].weights.length;
                for (let weightIndex = 0; weightIndex < weightCount; weightIndex++) {

                    const weightOnZ = this.deltaWeightOnZ(layerIndex, weightIndex);

                    const weightOnCost = weightOnZ * zOnActivity * activityOnCost;

                    console.log(`Effect of increasing weight of layer[${layerIndex}].nodes[${nodeIndex}].weights[${weightIndex}] on error: ${weightOnCost}`);

                    manipulationMatrix.addWeight(weightOnCost, layerIndex, nodeIndex, weightIndex);

                    if (layerIndex > 0) {
                        const prevNodeIndex = weightIndex;
                        const prevActivityOnZ = this.deltaPrevActivityOnZ(layerIndex, nodeIndex, prevNodeIndex);
                        const prevActivityOnCost = prevActivityOnZ * zOnActivity * activityOnCost;

                        memo.add(prevActivityOnCost, layerIndex - 1, prevNodeIndex);
                    }
                }
            }
        }
    }
}