import {NNNetwork} from "./structures/Network";
import {Backpropagation} from "./math/Backpropagation";
import {ManipulationMatrix} from "./math/Manipulation";

const main = () => {
    const network = new NNNetwork();
    network.init(2, [2, 2, 2], [1, 1, 1], [3, 3, 3]);

    let inputs = []
    let expectedOutputs = [];

    for (let i = 0; i < 1000; i++) {
        const first = Math.round(Math.random() * 100) / 100;
        const second = Math.round(Math.random() * 100) / 100;

        inputs.push([first, second]);
        expectedOutputs.push([Math.abs(1 - first), Math.abs(1 - second)]);
    }

    for (let epoch = 0; epoch < 100; epoch++) {
        console.log(`======== Starting Epoch # ${(epoch + 1)} ============`);

        let epochCost = 0;
        for (let batch = 0; batch < 100; batch++) {
            const manipulation = new ManipulationMatrix(network.export());

            for (let item = 0; item < 10; item++) {
                const output = network.calculate(inputs[batch * 10 + item]);
                const backprop = new Backpropagation(network.export(), expectedOutputs[batch * 10 + item]);

                backprop.execute(manipulation);

                epochCost += backprop.totalCost;
            }

            manipulation.apply(network, .5, .5);
        }

        console.log(`Total cost of epoch: ${epochCost}`);
        console.log(`Completed Epoch #${(epoch + 1)} `);
    }


    const testInput2 = [1,0];
    const testExpectedOutput2 = [0,1];
    const output2 = network.calculate(testInput2);
    console.log("==== Test ===");
    console.log("Output: " + output2);
    console.log("Expected Output: " + testExpectedOutput2);
    console.log("=============");
    const testInput3 = [0,1];
    const testExpectedOutput3 = [1,0];
    const output3 = network.calculate(testInput3);
    console.log("==== Test ===");
    console.log("Output: " + output3);
    console.log("Expected Output: " + testExpectedOutput3);
    console.log("=============");
    const testInput4 = [.5,.75];
    const testExpectedOutput4 = [.5,.25];
    const output4 = network.calculate(testInput4);
    console.log("==== Test ===");
    console.log("Output: " + output4);
    console.log("Expected Output: " + testExpectedOutput4);
    console.log("=============");


    // for (let iteration = 0; iteration < inputs.length; iteration++) {
    //     const output = network.calculate(inputs[iteration]);
    //     console.log(`======= Iteration #${iteration} =======`);
    //     console.log(`Inputs: ${inputs[iteration]}`);
    //     console.log(`Got Outputs: ${output}`);
    //     console.log(`Expected Outputs: ${expectedOutputs[iteration]}`);
    //     const backProp = new Backpropagation(network.export(), expectedOutputs[iteration])
    //     console.log(`Cost Array: ${backProp.costArray}`)
    //     console.log(`Total Array: ${backProp.totalCost}`)
    // }
}

main();

/**
 * https://github.com/ResoluteError/ts-neural-network
 * https://medium.com/@douglasreiser/building-a-deep-neural-network-from-scratch-in-typescript-9028903c15f1
 * At:
 * MNIST Digit Dataset
 */
