package com.azz.ai.activation;

// tanh(x)=(e^(x) - e^(-x))/(e^(x) + e^(-x))

public class TanH implements ActivationFunction {
    @Override
    public double output(double x) {
        return Math.tanh(x);
    }

    @Override
    public double outputDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
}