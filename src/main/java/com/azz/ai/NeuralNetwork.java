package com.azz.ai;

import com.azz.ai.activation.*;
import com.azz.ai.data.MLData;
import com.azz.ai.data.MLDataSet;
import com.azz.ai.layer.*;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


@Slf4j
public class NeuralNetwork {

    private final int inputSize;
    private final int[] hiddenSize;
    private final int outputSize;
    private ActivationFunction activationFunction;
    private final double learningRate;
    private final double momentum;
    private Layer inputLayer;
    private List<Layer> hiddenLayer;
    private Layer outputLayer;

    @Builder(builderMethodName = "config")
    private NeuralNetwork(int inputSize, int[] hiddenSize, int outputSize, double learningRate, double momentum,
                          ActivationFunctionType activationFunctionType) {
        log.info("Neural Network Initialization: START");
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.setActivationFunction(activationFunctionType);
        this.buildLayers();
        log.info("Neural Network Initialization: COMPLETED");
    }

    public void train(MLDataSet set, int epoch) {
        log.info("Training Started");
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(set.getData());

            for (MLData datum : set.getData()) {
                forward(datum.getInputs());
                backward(datum.getTargets());
            }
        }
        log.info("Training Finished.");
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[outputLayer.getNeurons().size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeurons().get(i).getOutput();
        }
        log.info("Input : {} Predicted : {}", Arrays.toString(inputs), Arrays.toString(output));
        return output;
    }

    private void setActivationFunction(ActivationFunctionType activationFunctionType) throws IllegalArgumentException {
        switch (Objects.requireNonNullElse(activationFunctionType, ActivationFunctionType.SIGMOID)) {
            case SIGMOID:
                this.activationFunction = new Sigmoid();
                break;
            case LEAKY_RELU:
                this.activationFunction = new LeakyReLu();
                break;
            case TANH:
                this.activationFunction = new TanH();
                break;
            case SWISH:
                this.activationFunction = new Swish();
                break;
            default:
                throw new IllegalArgumentException("Unknown activation function type: " + activationFunctionType);
        }
    }

    private void buildLayers() {
        this.inputLayer = new InputLayer(this.inputSize);
        this.hiddenLayer = IntStream.range(0, this.hiddenSize.length)
                .mapToObj(row -> new HiddenLayer(row, this.hiddenSize[row], this.inputLayer.getNeurons(), activationFunction))
                .collect(Collectors.toList());
        this.outputLayer = new OutputLayer(this.outputSize, this.hiddenLayer.getLast().getNeurons(), activationFunction);
    }


    private void backward(double[] targets) {
        IntStream.range(0, outputSize).forEach(itter -> outputLayer.getNeurons().get(itter).calculateGradient(targets[itter]));
        hiddenLayer.forEach(layer -> {
            layer.getNeurons().forEach(Neuron::calculateGradient);
            layer.getNeurons().forEach(neuron -> neuron.updateConnections(learningRate, momentum));
        });
        outputLayer.getNeurons().forEach(neuron -> neuron.updateConnections(learningRate, momentum));
    }

    private void forward(double[] inputs) {
        IntStream.range(0, inputSize).forEach(itter -> inputLayer.getNeurons().get(itter).setOutput(inputs[itter]));
        hiddenLayer.forEach(layer -> layer.getNeurons().forEach(Neuron::calculateOutput));
        outputLayer.getNeurons().forEach(Neuron::calculateOutput);
    }
}
