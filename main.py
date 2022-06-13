
# import OpenCV, Numpy, and Tensorflow
import cv2
import numpy as np
import tensorflow as tf

# Set up the TFLite Interpreter and some utility variables
interpreter = tf.lite.Interpreter(model_path='quant-default-1654829708.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# Initialize the webcam videostream
cap = cv2.VideoCapture(2)

# Check that the webcam videostream was opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Enter processing loop, only exit on press of key 'q'
while True:

    # Fetch frame from webcam
    ret, frame = cap.read()

    # Check that frame was returned
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Utility image variables
    width = frame.shape[1]
    height = frame.shape[0]
    scale = 2

    # Crop a center square from the image
    image = frame[height//2 - scale*input_height : height//2 + scale*input_height, width//2 - scale*input_width : width//2 + scale*input_width, :]

    # Resize to the model's input dimensions
    image = cv2.resize(image, (input_width, input_height))

    # Save for displaying later
    display = image.copy()

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Switch datatype to float for processing
    image = image.astype(float)

    # Scale by factor of 3 around 128
    image = (image - 128) * 3 + 128

    # Clip values between 0 and 255
    image = np.clip(image, 0, 255)

    # Switch datatype back to uint8
    image = image.astype(np.uint8)

    # Add dimensions to front and back
    image = np.expand_dims(image, axis=[0, -1]).astype(np.float32)

    # Set input tensor to tflite interpreter
    interpreter.set_tensor(input_index, image)

    # Make an inference
    interpreter.invoke()

    # Fetch the inference
    inference = np.argmax(interpreter.get_tensor(output_index)[0])

    # Scale up for user visibility
    display = cv2.resize(display, (480, 480))

    # Put inference on display
    display = cv2.putText(display, str(inference), (0, 480), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)

    # Display feed with inference
    cv2.imshow('display', display)

    # Check for press of key 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources allocated to webcam videostream
cap.release()

# Release resources allocated to cv2 windows
cv2.destroyAllWindows()
