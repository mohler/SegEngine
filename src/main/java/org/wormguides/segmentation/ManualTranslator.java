package org.wormguides.segmentation;

import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class ManualTranslator implements Translator<Image, CategoryMask> {

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToTensor());
        pipeline.add(new Normalize(
                new float[]{0.485f, 0.456f, 0.406f},
                new float[]{0.229f, 0.224f, 0.225f}
        ));
        
        NDArray array = input.toNDArray(ctx.getNDManager());
        NDList result = pipeline.transform(new NDList(array));
        return new NDList(result.get(0).expandDims(0));
    }

    @Override
    public CategoryMask processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities4D = list.get(0);
        NDArray probabilities = probabilities4D.squeeze(0);
        NDArray argMax = probabilities.argMax(0);
        
        // Move to CPU and get the Java array
        NDArray cpuArgMax = argMax.toDevice(Device.cpu(), false);
        int[] flat = cpuArgMax.toType(DataType.INT32, false).toIntArray();

        // 1. THE DICTIONARY
        List<String> classes = Arrays.asList(
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        );

        // 2. THE REPORT (This is what you've been missing)
        // We use a Set to find unique IDs so we don't print "dog" 94,000 times.
        Set<Integer> uniqueIds = new TreeSet<>();
        for (int val : flat) {
            uniqueIds.add(val);
        }

        System.out.println("==========================================");
        System.out.print("V100 DETECTED: ");
        for (int id : uniqueIds) {
            if (id > 0 && id < classes.size()) { // Skip background
                System.out.print("[" + classes.get(id).toUpperCase() + "] ");
            }
        }
        System.out.println("\n==========================================");

        // 3. RECONSTRUCT FOR CYTOSHOW
        long[] shape = cpuArgMax.getShape().getShape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        int[][] mask2d = new int[height][width];
        for (int i = 0; i < height; i++) {
            System.arraycopy(flat, i * width, mask2d[i], 0, width);
        }

        return new CategoryMask(classes, mask2d);
    }

    @Override
    public Batchifier getBatchifier() {
        return null; 
    }
}