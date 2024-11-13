package com.azz.ai.activation;

public interface ActivationFunction {

    double output(double x);

    double outputDerivative(double x);
}
