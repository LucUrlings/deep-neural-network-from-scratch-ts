import {NNNetworkObj} from "../structures/Network";

export class backpropogation {
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
}