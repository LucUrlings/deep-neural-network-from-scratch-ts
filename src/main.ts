import {NNNetwork} from "./structures/Network";
import {backpropogation} from "./math/backpropogation";

const main = () => {
    const network = new NNNetwork();
    network.init(2,[2,2,2],[1,1,1],[3,3,3])
    // console.log(JSON.stringify(network.export(), null, 2))

    let inputs = [[1,0],[0,0]]
    let expectedOutputs = [[0,1],[1,1]]

    for (let iteration = 0; iteration < inputs.length; iteration++) {
        const output = network.calculate(inputs[iteration]);
        console.log(`======= Iteration #${iteration} =======`);
        console.log(`Inputs: ${inputs[iteration]}`);
        console.log(`Got Outputs: ${output}`);
        console.log(`Expected Outputs: ${expectedOutputs[iteration]}`);
        const backProp = new backpropogation(network.export(), expectedOutputs[iteration])
        console.log(`Cost Array: ${backProp.costArray}`)
        console.log(`Total Array: ${backProp.totalCost}`)
    }
}

main();

/**
 * https://medium.com/@douglasreiser/building-a-deep-neural-network-from-scratch-in-typescript-9028903c15f1
 * At:
 * Derivatives for propagating backwards
 */
