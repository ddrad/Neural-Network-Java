package com.azz.ai.connection;

import com.azz.ai.Neuron;
import com.azz.ai.util.RandomGenerator;
import lombok.Getter;
import lombok.Setter;

import java.util.UUID;

@Getter
@Setter
public class Connection {


    private UUID connectionId;
    private Neuron from;
    private Neuron to;
    private double synapticWeight;
    private double synapticWeightDelta;

    public Connection(Neuron from, Neuron to) {
        this.connectionId = UUID.randomUUID();
        this.from = from;
        this.to = to;
        this.synapticWeight = RandomGenerator.randomValue(-2, 2);
    }

    public void updateSynapticWeight(double synapticWeight) {
        this.synapticWeight += synapticWeight;
    }

}
