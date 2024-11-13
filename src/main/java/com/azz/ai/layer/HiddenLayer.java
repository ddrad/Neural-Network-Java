package com.azz.ai.layer;

import com.azz.ai.Neuron;
import com.azz.ai.activation.ActivationFunction;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;



@Getter
public class HiddenLayer implements Hidden {

    private int row;
    private List<Neuron> neurons = new ArrayList<Neuron>();

    public HiddenLayer(int row, int size, List<Neuron> pre, ActivationFunction activationFunction) {
        this.row = row;
        for (int i = 0; i < size; i++) {
            this.getNeurons().add(new Neuron(pre, activationFunction));
        }
    }
}
