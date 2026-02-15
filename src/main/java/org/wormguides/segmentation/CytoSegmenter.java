package org.wormguides.segmentation;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ij.ImagePlus;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * CytoSegmenter: GPU-accelerated deep learning segmentation engine.
 * Optimized for Java 8 and NVIDIA Tesla V100 via DJL 0.23.0.
 */
public class CytoSegmenter {

    static ZooModel<Image, CategoryMask> model;
    private static final String MODEL_NAME = "resnet50_desktop.pt";


    /**
     * Initializes the DeepLabV3 model.
     */
    public static synchronized void init() throws ModelException, IOException {
        if (model != null) return;

        // DIRECT PATH: This bypasses the 'ModelNotFoundException' entirely.
        // Note: DJL likes the folder path for optModelUrls, then the filename for optModelName.
        String modelFolder = "file:///C:/models/"; 
        String modelFileName = "resnet101_v100"; // Matches resnet101_v100.pt

        System.out.println("--- Initializing AI Engine (Direct V100 Trace) ---");
        System.out.println("Loading local file: " + modelFolder + modelFileName + ".pt");

        Criteria<Image, CategoryMask> criteria = Criteria.builder()
                .setTypes(Image.class, CategoryMask.class)
                .optModelUrls(modelFolder)
                .optModelName(modelFileName)
                .optEngine("PyTorch")
                .optDevice(Device.gpu(0)) // Force Tesla V100
                .optTranslator(new ManualTranslator())
                .build();

        // This now loads the file from your C: drive, NOT the internet.
        model = ModelZoo.loadModel(criteria);
        System.out.println("SUCCESS: AI Engine Bound to Tesla V100 via Local Artifact.");
    }
    
    /**
     * STEP 3 FIX: Async Preload
     * Call this at application startup to warm up the GPU.
     */
    public static void preload() {
        new Thread(() -> {
            try {
                init();
            } catch (Exception e) {
                System.err.println("Background model load failed: " + e.getMessage());
            }
        }).start();
    }

    public static ImagePlus segment(ImagePlus inputImp) {
        if (inputImp == null) return null;

        try {
            init();

            ImageProcessor ip = inputImp.getProcessor();
            BufferedImage buffImg = ip.getBufferedImage();
            Image img = ImageFactory.getInstance().fromImage(buffImg);

            try (Predictor<Image, CategoryMask> predictor = model.newPredictor()) {
                CategoryMask mask = predictor.predict(img);

                // Get the raw mask image (Grayscale integers)
                Image maskImage = mask.getMaskImage(img);
                BufferedImage rawBuff = (BufferedImage) maskImage.getWrappedImage();
                ImagePlus rawImp = new ImagePlus("Raw_Mask", rawBuff);
                
                // STEP 2 FIX: Return a color-mapped visualization
                return applyColorMap(rawImp);
            }

        } catch (Exception e) {
            System.err.println("CRITICAL ERROR during segmentation inference:");
            e.printStackTrace();
            return null;
        }
    }
    
     /* Helper to convert raw integer masks into a visible heatmap.
     * Uses the native ColorProcessor constructor (int width, int height, int[] pixels).
     */
    private static ImagePlus applyColorMap(ImagePlus rawImp) {
        ImageProcessor ip = rawImp.getProcessor();
        int width = ip.getWidth();
        int height = ip.getHeight();
        
        // 1. Create the integer array that ColorProcessor expects
        int[] colorPixels = new int[width * height];
        
        // Define colors (0=Transparent, 1=Red, 12=Blue, etc.)
        // We use .getRGB() to get the packed integer (0xAARRGGBB)
        int[] palette = {
            0x00000000, // Class 0: Transparent
            Color.RED.getRGB(), Color.GREEN.getRGB(), Color.BLUE.getRGB(), Color.CYAN.getRGB(), 
            Color.MAGENTA.getRGB(), Color.YELLOW.getRGB(), Color.ORANGE.getRGB(), Color.PINK.getRGB(), new Color(128,0,0).getRGB(),
            new Color(0,128,0).getRGB(), new Color(0,0,128).getRGB(), new Color(128,128,0).getRGB(), new Color(128,0,128).getRGB(), new Color(0,128,128).getRGB(),
            new Color(192,192,192).getRGB(), new Color(128,128,128).getRGB(), new Color(153,51,0).getRGB(), new Color(51,153,0).getRGB(), new Color(51,0,153).getRGB(), new Color(0,153,153).getRGB()
        };
        
        // 2. Map the mask IDs (0, 1, 2...) to RGB Integers
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int classId = ip.getPixel(x, y);
                if (classId >= 0 && classId < palette.length) {
                    colorPixels[y * width + x] = palette[classId];
                }
            }
        }
        
        // 3. Call the EXACT constructor you found
        ColorProcessor cp = new ColorProcessor(width, height, colorPixels);
        
        return new ImagePlus("Segmented_Overlay", cp);
    }
}