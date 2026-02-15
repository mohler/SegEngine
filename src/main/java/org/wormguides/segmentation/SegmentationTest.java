package org.wormguides.segmentation;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import javax.imageio.ImageIO;

public class SegmentationTest {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        System.out.println("=== 1. Starting Dual-Output Test ===");
        
        String url = "https://resources.djl.ai/images/dog_bike_car.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        CytoSegmenter.init();

        try (ai.djl.inference.Predictor<Image, CategoryMask> predictor = CytoSegmenter.model.newPredictor(new ManualTranslator())) {
            CategoryMask mask = predictor.predict(img);
            
            int[][] rawData = mask.getMask(); 
            int height = rawData.length;
            int width = rawData[0].length;
            
            // --- OUTPUT 1: HIGH-VISIBILITY NEON (FOR HUMANS) ---
            BufferedImage visual = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            
            // --- OUTPUT 2: RAW QUANTITATIVE (FOR CYTOSHOW) ---
            BufferedImage raw = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            byte[] rawPixels = ((DataBufferByte) raw.getRaster().getDataBuffer()).getData();
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int classId = rawData[y][x];
                    
                    // Populate Raw Image (Science Mode)
                    rawPixels[y * width + x] = (byte) classId;

                    // Populate Visual Image (Neon Mode)
                    int r = 0, g = 0, b = 0;
                    if (classId > 0) {
                        r = Math.min(classId * 20, 255); 
                        g = Math.max(255 - (classId * 20), 0);
                        b = 255; 
                    }
                    int rgb = (r << 16) | (g << 8) | b;
                    visual.setRGB(x, y, rgb);
                }
            }

            // Save both
            ImageIO.write(visual, "png", new File("test_segmentation_visual.png"));
            ImageIO.write(raw, "png", new File("test_segmentation_raw.png"));
            
            System.out.println("=== SUCCESS ===");
            System.out.println("Visual Map: test_segmentation_visual.png");
            System.out.println("Quantitative Map: test_segmentation_raw.png");
        }
    }
}