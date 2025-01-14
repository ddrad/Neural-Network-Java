package com.azz.ai.data;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class MLData {

    private double[] inputs;
    private double[] targets;

    public MLData(double[] inputs, double[] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

}
