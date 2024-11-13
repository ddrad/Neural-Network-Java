package com.azz.ai.layer;

import com.azz.ai.Neuron;
import com.azz.ai.activation.ActivationFunction;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class OutputLayer implements Layer {

    public List<Neuron> neurons = new ArrayList<>();

    public OutputLayer(int size, List<Neuron> pre, ActivationFunction activationFunction) {
        for (int i = 0; i < size; i++) {
            this.getNeurons().add(new Neuron(pre, activationFunction));
        }
    }

    /*
        public List<Neuron> neurons = new ArrayList<Neuron>();

    // Constructor for the hidden and output layer
    public Layer(int size, List<Neuron> pre, ActivationFunction activationFunction) {
        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(pre, activationFunction));
        }
    }


    // Constructor for the input layer
    public Layer(int inputSize) {
        for (int i = 0; i < inputSize; i++) {
           this.neurons.add(new Neuron());
        }
    }
     */
}
