import time
import cv2

from gpiozero import (
    AngularServo,
    Button,
    DigitalInputDevice,
    DigitalOutputDevice
)
from ultralytics import YOLO


# ============================================================
# GPIO PIN CONFIGURATION
# ============================================================

# LEDs
GREEN_LED_PIN = 24
RED_LED_PIN = 16
YELLOW_LED_PIN = 25

# Servo
SERVO_PIN = 18

# Conveyor relays
MOTOR_FORWARD_PIN = 23
MOTOR_REVERSE_PIN = 12

# Input devices
START_BUTTON_PIN = 17
STOP_BUTTON_PIN = 5
PHOTO_SENSOR_PIN = 22


# ============================================================
# SYSTEM SETTINGS
# ============================================================

MODEL_PATH = "best.pt"

CONFIDENCE_THRESHOLD = 0.50

# Conveyor movement time after successful detection
SORTING_RUN_TIME = 1.0

# Conveyor reverse time for rejected objects
REJECT_REVERSE_TIME = 2.0

# Maximum camera scanning time
SCAN_TIMEOUT = 2.0

# Time given to the servo before conveyor movement
SERVO_MOVE_DELAY = 0.5

# Servo positions
CAN_SERVO_ANGLE = 0
BOTTLE_SERVO_ANGLE = 65

# Logitech camera index
CAMERA_INDEX = 0

# Set True when using Raspberry Pi desktop and you want
# to display the camera window.
SHOW_CAMERA = True

# Most relay modules switch ON when GPIO is HIGH.
# Change this to False if your relay activates when GPIO is LOW.
RELAY_ACTIVE_HIGH = True


# ============================================================
# OUTPUT DEVICES
# ============================================================

green_led = DigitalOutputDevice(
    GREEN_LED_PIN,
    active_high=True,
    initial_value=False
)

red_led = DigitalOutputDevice(
    RED_LED_PIN,
    active_high=True,
    initial_value=False
)

yellow_led = DigitalOutputDevice(
    YELLOW_LED_PIN,
    active_high=True,
    initial_value=False
)

motor_forward_relay = DigitalOutputDevice(
    MOTOR_FORWARD_PIN,
    active_high=RELAY_ACTIVE_HIGH,
    initial_value=False
)

motor_reverse_relay = DigitalOutputDevice(
    MOTOR_REVERSE_PIN,
    active_high=RELAY_ACTIVE_HIGH,
    initial_value=False
)


# ============================================================
# INPUT DEVICES
# ============================================================

# pull_up=False means:
# normal state = LOW
# pressed state = HIGH
start_button = Button(
    START_BUTTON_PIN,
    pull_up=False,
    bounce_time=0.1
)

stop_button = Button(
    STOP_BUTTON_PIN,
    pull_up=False,
    bounce_time=0.1
)

# Photo sensor:
# normal = LOW
# object detected = HIGH
photo_sensor = DigitalInputDevice(
    PHOTO_SENSOR_PIN,
    pull_up=False
)


# ============================================================
# SERVO SETUP
# ============================================================

servo = AngularServo(
    SERVO_PIN,
    min_angle=0,
    max_angle=90,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025
)

servo.angle = CAN_SERVO_ANGLE


# ============================================================
# LOAD YOLO MODEL AND CAMERA
# ============================================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Opening Logitech camera...")
camera = cv2.VideoCapture(CAMERA_INDEX)

# Optional camera settings
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open camera. Try changing CAMERA_INDEX to 1."
    )


# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def stop_conveyor():
    """
    Stop both forward and reverse relays.

    Both relays must never be active at the same time.
    """
    motor_forward_relay.off()
    motor_reverse_relay.off()
    red_led.off()


def move_conveyor_forward():
    """Move conveyor forward and turn the red LED on."""
    motor_reverse_relay.off()
    time.sleep(0.05)

    motor_forward_relay.on()
    red_led.on()


def move_conveyor_reverse():
    """Move conveyor in reverse."""
    motor_forward_relay.off()
    time.sleep(0.05)

    motor_reverse_relay.on()


def wait_with_stop_check(duration):
    """
    Wait for a specific duration while continuously checking
    the Stop button.

    Returns:
        True  - movement completed normally
        False - Stop button was pressed
    """
    end_time = time.monotonic() + duration

    while time.monotonic() < end_time:
        if stop_button.is_pressed:
            stop_conveyor()
            return False

        time.sleep(0.02)

    return True


# ============================================================
# LED FUNCTIONS
# ============================================================

def flash_red_led_while_reversing(duration):
    """
    Reverse the conveyor while flashing the red LED.

    Used when the object is not a bottle or can,
    or when confidence is below 0.50.
    """
    move_conveyor_reverse()

    end_time = time.monotonic() + duration
    led_state = False
    last_flash_time = 0

    while time.monotonic() < end_time:
        if stop_button.is_pressed:
            stop_conveyor()
            return False

        current_time = time.monotonic()

        if current_time - last_flash_time >= 0.25:
            led_state = not led_state

            if led_state:
                red_led.on()
            else:
                red_led.off()

            last_flash_time = current_time

        time.sleep(0.02)

    stop_conveyor()
    return True


# ============================================================
# CAMERA AND YOLO FUNCTIONS
# ============================================================

def draw_detection(frame, box, class_name, confidence):
    """Draw the bounding box and label on the camera image."""
    x1, y1, x2, y2 = box.xyxy[0]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


def scan_object():
    """
    Scan the stopped object using YOLO.

    The function examines camera frames for up to SCAN_TIMEOUT
    seconds and remembers the highest-confidence result.

    Returns:
        ("bottle", confidence)
        ("can", confidence)
        (None, confidence)
    """
    yellow_led.on()

    scan_start_time = time.monotonic()

    best_valid_class = None
    best_valid_confidence = 0.0
    highest_confidence_seen = 0.0

    while time.monotonic() - scan_start_time < SCAN_TIMEOUT:
        if stop_button.is_pressed:
            yellow_led.off()
            return None, 0.0

        ret, frame = camera.read()

        if not ret:
            print("Cannot read camera frame")
            time.sleep(0.05)
            continue

        results = model.predict(
            source=frame,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id].lower().strip()

                highest_confidence_seen = max(
                    highest_confidence_seen,
                    confidence
                )

                print(
                    f"Detected: {class_name}, "
                    f"confidence: {confidence:.2f}"
                )

                draw_detection(
                    frame,
                    box,
                    class_name,
                    confidence
                )

                valid_class = class_name in ("bottle", "can")
                valid_confidence = confidence >= CONFIDENCE_THRESHOLD

                if (
                    valid_class
                    and valid_confidence
                    and confidence > best_valid_confidence
                ):
                    best_valid_class = class_name
                    best_valid_confidence = confidence

        if SHOW_CAMERA:
            cv2.imshow("RVM Camera", frame)

            # Press Q to shut down the complete program.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        # Stop scanning early after receiving a strong result.
        if best_valid_confidence >= 0.80:
            break

    yellow_led.off()

    if best_valid_class is not None:
        return best_valid_class, best_valid_confidence

    else:
        return None, highest_confidence_seen


# ============================================================
# OBJECT PROCESSING
# ============================================================

def process_detected_object():
    """
    Stop conveyor, scan object, set servo, and route object.
    """
    print("\nObject detected by photo sensor")

    stop_conveyor()
    time.sleep(0.2)

    detected_class, confidence = scan_object()

    if stop_button.is_pressed:
        return False

    if detected_class == "bottle":
        print(
            f"Bottle accepted: confidence {confidence:.2f}"
        )

        servo.angle = BOTTLE_SERVO_ANGLE
        time.sleep(SERVO_MOVE_DELAY)

        move_conveyor_forward()

        if not wait_with_stop_check(SORTING_RUN_TIME):
            return False

        stop_conveyor()

        # Return the sorting servo to its default position.
        servo.angle = CAN_SERVO_ANGLE

    elif detected_class == "can":
        print(
            f"Can accepted: confidence {confidence:.2f}"
        )

        servo.angle = CAN_SERVO_ANGLE
        time.sleep(SERVO_MOVE_DELAY)

        move_conveyor_forward()

        if not wait_with_stop_check(SORTING_RUN_TIME):
            return False

        stop_conveyor()

    else:
        print(
            "Object rejected: not a bottle/can "
            f"or confidence below {CONFIDENCE_THRESHOLD:.2f}"
        )

        if not flash_red_led_while_reversing(
            REJECT_REVERSE_TIME
        ):
            return False

    return True


def wait_until_sensor_clears():
    """
    Prevent the same object from triggering the sensor repeatedly.
    """
    print("Waiting for sensor area to clear...")

    while photo_sensor.value:
        if stop_button.is_pressed:
            stop_conveyor()
            return False

        time.sleep(0.05)

    time.sleep(0.2)
    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    system_running = False

    # Green LED represents power/system ready.
    green_led.on()

    # Safe initial state.
    stop_conveyor()
    yellow_led.off()
    servo.angle = CAN_SERVO_ANGLE

    print("\n======================================")
    print("Reverse Vending Machine Ready")
    print("======================================")
    print("Green LED: ON")
    print("Servo position: 0 degrees")
    print("Press START to begin")
    print("Press STOP to stop the conveyor")
    print("Press Q in the camera window to exit")
    print("======================================\n")

    while True:
        # Stop button always has priority.
        if stop_button.is_pressed:
            if system_running:
                print("STOP button pressed")

            system_running = False
            stop_conveyor()
            yellow_led.off()

            # Wait until the button is released.
            while stop_button.is_pressed:
                time.sleep(0.05)

        # Start the system only after Start button is pressed.
        if not system_running:
            stop_conveyor()

            if start_button.is_pressed:
                print("START button pressed")
                print("System started")

                servo.angle = CAN_SERVO_ANGLE
                time.sleep(0.3)

                system_running = True
                move_conveyor_forward()

                # Wait for button release to prevent repeated starts.
                while start_button.is_pressed:
                    if stop_button.is_pressed:
                        break

                    time.sleep(0.05)

            else:
                time.sleep(0.05)
                continue

        # Normal conveyor operation.
        if system_running:
            # Ensure conveyor is moving forward while waiting.
            if not motor_forward_relay.value:
                move_conveyor_forward()

            # Sensor becomes HIGH when an object arrives.
            if photo_sensor.value:
                successful = process_detected_object()

                if not successful or stop_button.is_pressed:
                    system_running = False
                    stop_conveyor()
                    continue

                # Wait until the object leaves the sensor.
                if not wait_until_sensor_clears():
                    system_running = False
                    continue

                # Continue running for the next object.
                if system_running:
                    print("Ready for next object\n")
                    move_conveyor_forward()

            time.sleep(0.02)


# ============================================================
# PROGRAM START AND CLEANUP
# ============================================================

try:
    main()

except KeyboardInterrupt:
    print("\nProgram stopped by user")

except Exception as error:
    print(f"\nSystem error: {error}")

finally:
    print("Shutting down safely...")

    stop_conveyor()

    green_led.off()
    red_led.off()
    yellow_led.off()

    servo.angle = CAN_SERVO_ANGLE
    time.sleep(0.3)

    camera.release()
    cv2.destroyAllWindows()

    servo.close()

    green_led.close()
    red_led.close()
    yellow_led.close()

    motor_forward_relay.close()
    motor_reverse_relay.close()

    start_button.close()
    stop_button.close()
    photo_sensor.close()

    print("GPIO and camera released")