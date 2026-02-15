package org.wormguides.segmentation;

import ai.djl.engine.Engine;
import ai.djl.Device;

public class GpuTest {
    public static void main(String[] args) {
        // 1. Get the actual Engine instance (this loads PyTorch)
        Engine engine = Engine.getInstance();
        
        System.out.println("--------------------------------------------------");
        System.out.println("DJL Version: " + Engine.getDjlVersion());
        System.out.println("Engine Name: " + engine.getEngineName());
        System.out.println("--------------------------------------------------");
        
        // 2. Ask the Engine explicitly how many GPUs it sees (Java 8 / DJL 0.23.0 compatible)
        int gpuCount = engine.getGpuCount();
        System.out.println("GPUs Available: " + gpuCount);
        
        if (gpuCount > 0) {
            // 3. Ask for the default device (usually GPU:0)
            Device device = engine.defaultDevice();
            System.out.println("SUCCESS: We are running on: " + device);
            System.out.println("You are ready for segmentation.");
        } else {
            System.err.println("WARNING: No GPU detected. We are falling back to CPU (Slow!)");
        }
    }
}