package com.azz.ai.layer;

import com.azz.ai.Neuron;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class InputLayer implements Layer {

    public List<Neuron> neurons = new ArrayList<Neuron>();

    public InputLayer(int inputSize) {
        for (int i = 0; i < inputSize; i++) {
            this.getNeurons().add(new Neuron());
        }
    }

}
