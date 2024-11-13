package com.azz.ai;

import com.azz.ai.activation.ActivationFunctionType;
import com.azz.ai.data.MLDataSet;

public class Main {

    private static final double[][] XOR_INPUT = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    private static final double[][] XOR_IDEAL = {
            {0},
            {1},
            {1},
            {0}
    };

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = NeuralNetwork.config()
                .inputSize(2)
                .hiddenSize(new int[]{10})
                .outputSize(1)
                .activationFunctionType(ActivationFunctionType.SIGMOID)
                .learningRate(0.01)
                .momentum(0.5)
                .build();

        MLDataSet dataSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);
        neuralNetwork.train(dataSet, 100000);

        neuralNetwork.predict(1, 1);
        neuralNetwork.predict(1, 0);
        neuralNetwork.predict(0, 1);
        neuralNetwork.predict(0, 0);
    }
}