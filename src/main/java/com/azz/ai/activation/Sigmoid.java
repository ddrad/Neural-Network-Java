package com.azz.ai.activation;

// f(x) = 1/(1+e^(-x))

public class Sigmoid implements ActivationFunction {
    @Override
    public double output(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double outputDerivative(double x) {
        return x * (1 - x);
    }
}