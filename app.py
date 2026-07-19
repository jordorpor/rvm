import cv2
from ultralytics import YOLO
# from gpiozero import AngularServo


# Load your trained model
model = YOLO("best.pt")


# Servo connected to GPIO 18
# servo = AngularServo(
#    18,
 #   min_angle=0,
  #  max_angle=90)

# Start servo at 0 degrees
# servo.angle = 0


# Open camera
camera = cv2.VideoCapture(0)


while True:

    # Get image from camera
    ret, frame = camera.read()

    if not ret:
        print("Cannot read camera")
        break


    # Detect objects
    results = model(frame)


    # Check detected objects
    for result in results:

        for box in result.boxes:

            # Get class number
            class_id = int(box.cls[0])

            # Get confidence
            confidence = float(box.conf[0])

            # Convert class number to class name
            class_name = result.names[class_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            print(class_name, confidence)


            # Only use detection above 70%
            if confidence > 0.1:
                
                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display class name and confidence
                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                if class_name.lower() == "can":
                    print("Can detected")
                    # servo.angle = 90

                elif class_name.lower() == "bottle":
                    print("Bottle detected")
                    # servo.angle = 0


    # Show camera image
    cv2.imshow("Camera", frame)


    # Press Q to stop
    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
# servo.close()

