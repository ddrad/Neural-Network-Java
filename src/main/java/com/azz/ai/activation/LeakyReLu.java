package com.azz.ai.activation;

// f(x) = max(0, x)
// or
//         {  x if x>= 0 }
//  f(x) =
//         { 0 if x < 0  }

public class LeakyReLu implements ActivationFunction {

    @Override
    public double output(double x) {
        return x >= 0 ? x : x * 0.01;
    }

    @Override
    public double outputDerivative(double x) {
        return x >= 0 ? 1 : 0.01;
    }
}
