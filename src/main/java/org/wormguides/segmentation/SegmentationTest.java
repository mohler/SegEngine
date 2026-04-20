package org.wormguides.segmentation;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.translate.TranslateException;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class SegmentationTest {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        System.out.println("=== 1. Starting Dual-Output Test ===");
        
        String url = "https://resources.djl.ai/images/dog_bike_car.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        ImagePlus imp = new ImagePlus("Loaded_From_URL", (BufferedImage) img.getWrappedImage());        //Here is where to create new modified img from img->imp->ip.methodOfChoice()->newImp->newBuffImp->newImg;
     // 1. Transform via ImageJ (always duplicate to preserve the original)
        ImageProcessor ip = imp.getProcessor().duplicate();
        ip.setInterpolationMethod(ImageProcessor.BILINEAR);

        // Titration modifications
//        ip.scale(0.5, 0.5);       // Test resolution/scaling
        ip.multiply(1.2);         // Test contrast/brightness
        ip.rotate(180.0);          // Test orientation

        // 2. Convert back to ImagePlus, then to standard Java BufferedImage
        ImagePlus modifiedImp = new ImagePlus("Titration_Step", ip);
        BufferedImage buffModImg = modifiedImp.getBufferedImage();

        // 3. Ingest directly into DJL (Bypassing raw byte extraction completely)
        ImageFactory factory = ImageFactory.getInstance();
        Image modImg = factory.fromImage(buffModImg);
        
        
        CytoSegmenter.init();

        try (ai.djl.inference.Predictor<Image, CategoryMask> predictor = CytoSegmenter.model.newPredictor(new ManualTranslator())) {
            CategoryMask mask = predictor.predict(modImg);
            
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
            File tsVisPng = new File("test_segmentation_visual.png");
            File tsRawPng = new File("test_segmentation_raw.png");
            ImageIO.write(visual, "png", tsVisPng); 
            ImageIO.write(raw, "png", tsRawPng);
            
            System.out.println("=== SUCCESS ===");
            System.out.println("Visual Map: "+tsVisPng.getAbsolutePath());
            System.out.println("Quantitative Map: "+tsVisPng.getAbsolutePath());
        }
    }
}