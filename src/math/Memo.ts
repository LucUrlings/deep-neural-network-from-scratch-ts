export class Memo {
    activationOnCost: number[][];

    constructor() {
        this.activationOnCost = [];
    }

    init(layerCount: number) {
        this.activationOnCost = [];
        for (let i = 0; i < layerCount; i++) {
            this.activationOnCost.push([]);
        }
    }

    add(value: number, layerIndex: number, nodeIndex: number) {
        if (!this.has(layerIndex, nodeIndex)) {
            this.activationOnCost[layerIndex][nodeIndex] = value;
        } else {
            this.activationOnCost[layerIndex][nodeIndex] += value;
        }
    }

    has(layerIndex: number, nodeIndex: number) {
        return this.activationOnCost[layerIndex][nodeIndex] != null;
    }

    get(layerIndex: number, nodeIndex: number) {
        if (!this.has(layerIndex, nodeIndex)) {
            throw new Error(`No memo for activityOnCost[${layerIndex}][${nodeIndex}]`)
        }
        return this.activationOnCost[layerIndex][nodeIndex]
    }
}