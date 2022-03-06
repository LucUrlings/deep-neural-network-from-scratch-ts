import {NNLayer} from "./Layer";

export interface NNNetworkObj {
    layers: NNLayer[];
    input?: number[];
}

export class NNNetwork implements NNNetworkObj {
    layers: NNLayer[];
    input?: number[];

    constructor() {
        this.layers = [];
    }

    init(inputLength: number, nodesInLayer: number[], weightsInLayer: number[], biasInLayer: number[]): void {
        this.layers = [];
        if (nodesInLayer.length !== weightsInLayer.length || nodesInLayer.length !== biasInLayer.length) {
            throw new Error("Initializing NN failed - array lengths don't match");
        }

        for (let i = 0; i < nodesInLayer.length; i++) {
            let prevNodes;
            if (i === 0) {
                prevNodes = inputLength;
            } else {
                prevNodes = nodesInLayer[i - 1];
            }
            const newLayer = new NNLayer();
            newLayer.init(nodesInLayer[i], prevNodes, weightsInLayer[i], biasInLayer[i]);
            this.layers.push(newLayer);
        }
    }

    load(networkData: NNNetworkObj): void {
        this.layers = [];
        for (let i = 0; i < networkData.layers.length; i++) {
            const newLayer = new NNLayer();
            newLayer.load(networkData.layers[i]);
            this.layers.push(newLayer);
        }
    }

    export(): NNNetworkObj {
        return {
            layers: this.layers,
            input: this.input
        }
    }

    calculate(inputValues: number[]): number[] {
        this.input = inputValues;
        for (let i = 0; i < this.layers.length; i++) {
            inputValues = this.layers[i].calculateValues(inputValues);
        }
        return inputValues;
    }
}