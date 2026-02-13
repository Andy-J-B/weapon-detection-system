import cv2
import numpy as np
import sys
import os

# --- Constants (Must match C++ exactly) ---
MODEL_PATH = "server/best.onnx"  # Adjust if your path is different
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2  # Low threshold for testing
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Class IDs (Based on your training: 0=Human, 1=Gun)
CLASS_ID_HUMAN = 0
CLASS_ID_GUN = 1
CLASSES = ["Human", "Gun"]


def run_yolo_inference(image_path=None):
    print("------------------------------------------------")
    print("🚀 Starting Simulation of C++ Inference Pipeline")
    print("------------------------------------------------")

    # 1. Load Image or Generate Dummy
    if image_path and os.path.exists(image_path):
        print(f"📄 Loading image: {image_path}")
        image = cv2.imread(image_path)
    else:
        print("⚠️  No image provided or file not found.")
        print("🎨 Generating a blank (black) test image...")
        image = np.zeros((640, 640, 3), dtype=np.uint8)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Make sure you ran the training script and exported to ONNX first.")
        return

    print(f"🧠 Loading ONNX model from: {MODEL_PATH}")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    # Set backend to OpenCV (Simulating your C++ setup)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 3. Pre-process (blobFromImage)
    #    matches: cv::dnn::blobFromImage(image, blob, 1.0/255.0, size, scalar, true, false);
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # 4. Forward Pass
    print("⚡ Running forward pass...")
    outputs = net.forward()

    # YOLOv5 Output Shape: (1, 25200, 5 + NumClasses)
    # Usually (1, 25200, 7) for 2 classes (x, y, w, h, conf, p(human), p(gun))
    # We strip the first dimension [0]
    predictions = outputs[0]

    print(f"📊 Output Shape: {predictions.shape}")

    # 5. Post-Process (Unwrapping)
    class_ids = []
    confidences = []
    boxes = []

    rows = predictions.shape[0]

    # Calculate scaling factors to map 640x640 back to original image size
    x_factor = image.shape[1] / INPUT_WIDTH
    y_factor = image.shape[0] / INPUT_HEIGHT

    for r in range(rows):
        row = predictions[r]
        confidence = row[4]

        # Filter weak detections
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get class with max score
            class_id = np.argmax(classes_scores)
            max_class_score = classes_scores[class_id]

            # Check if class score is high enough
            if max_class_score > SCORE_THRESHOLD:
                x, y, w, h = row[0], row[1], row[2], row[3]

                # Convert from Center-X, Center-Y to Top-Left Corner
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 6. Non-Maximum Suppression (NMS)
    #    matches: cv::dnn::NMSBoxes(...)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    print(f"🔍 Raw detections: {len(boxes)}")
    print(f"🧹 Detections after NMS: {len(indexes)}")

    weapon_detected = False

    if len(indexes) > 0:
        for i in indexes.flatten():
            cls_id = class_ids[i]
            conf = confidences[i]
            box = boxes[i]

            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"ID {cls_id}"
            color = (0, 255, 0)  # Green for Human

            if cls_id == CLASS_ID_GUN:
                weapon_detected = True
                label = f"⚠️ WEAPON: {label}"
                color = (0, 0, 255)  # Red for Weapon
                print(f"🔴 DETECTION: GUN Found! Confidence: {conf:.2f}")
            else:
                print(f"🟢 DETECTION: Human Found. Confidence: {conf:.2f}")

            # Draw box for visual verification
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                f"{label} {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    if weapon_detected:
        print("\n🚨 FINAL DECISION: WEAPON DETECTED 🚨")
    else:
        print("\n✅ FINAL DECISION: SAFE (No Weapon)")

    # Show result window (Optional, if running locally)
    # cv2.imshow("Inference Simulation", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the output image to verify
    output_filename = "inference_result.jpg"
    cv2.imwrite(output_filename, image)
    print(f"💾 Result saved to {output_filename}")


if __name__ == "__main__":
    # You can pass an image path as an argument:
    # python test_inference.py my_gun_image.jpg
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_yolo_inference(img_arg)
