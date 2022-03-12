import {NNNode} from "./Node";

export interface NNLayerObj {
    nodes: NNNode[];
}

export class NNLayer implements NNLayerObj {
    nodes: NNNode[];

    constructor() {
        this.nodes = [];
    }

    init(breadth: number, inputNodes: number, weightRange: number, biasRange: number): void {
        this.nodes = [];
        for (let i = 0; i < breadth; i++) {
            const newNode = new NNNode();
            newNode.init(inputNodes, weightRange, biasRange);
            this.nodes.push(newNode);
        }
    }

    load(layer: NNLayerObj): void {
        this.nodes = [];
        for (let i = 0; i < layer.nodes.length; i++) {
            const newNode = new NNNode();
            newNode.load(layer.nodes[i]);
            this.nodes.push(newNode);
        }
    }

    export(): NNLayerObj {
        return {
            nodes: this.nodes
        }
    }

    calculateValues(inputValues: number[]): number[] {
        const resultArray = [];
        for (const node of this.nodes) {
            resultArray.push(node.calculateValue(inputValues));
        }

        return resultArray;
    }

    get nodeActivity() {
        const result = [];
        for (const node of this.nodes) {
            result.push(node.activity);
        }

        return result;
    }
}
