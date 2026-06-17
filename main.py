import cv2
import serial
import duckdb
import re
import string
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# Initialize client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="ezURGt6TbLiOTdMOT4Fa"
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# # Configure streaming options
config = StreamConfig(
    stream_output=["output_image"],  # Get video back with annotations
    data_output=["predictions","open_ai"],     # Get prediction data via datachannel
    processing_timeout=3600,             # 60 minutes
    requested_plan="webrtc-gpu-medium",  # Options: webrtc-gpu-small, webrtc-gpu-medium, webrtc-gpu-large
    requested_region="us"                # Options: us, eu, ap
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="text-recognition",
    workspace="obbierro-lastla",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    if data.get("predictions").get("image").get("width") != None:  # Check if predictions are present
        print(f"Time: {datetime.now()} -> Frame {metadata.frame_id}: {data}")
        plat_nomor = str(data.get("open_ai")[0]['output'])
        plat_nomor = plat_nomor[:9]
        print(f"PLAT NOMOR: {plat_nomor}")

        if(len(plat_nomor) > 8):
            check_valid(plat_nomor)

def check_valid(plat_nomor):
    print(f'plat_nomor check valid: {plat_nomor}')
    ser = serial.Serial('COM6', 115200, timeout=2)  # Open serial port
    RFID_tag = ser.readline().decode('utf-8').strip()
    print(f"Received data from ESP32: {RFID_tag}")  # Print received data
    compared_plate_number = open_database(RFID_tag)
    print(f"Compared plate number from database: {compared_plate_number}")
    sama = re.match(plat_nomor, compared_plate_number).group()
    print(f"Matched plate number: {sama} |Length: {len(sama)}")
    if(len(re.match(plat_nomor, compared_plate_number).group()) > 8):
        print("Access granted")
        ser.write(b'1')  # Send '1' to ESP32

def open_database(RFID_Tag=None):
    csv_file = "database.csv"
    RFID_Tag = RFID_Tag if RFID_Tag is not None else ""
    query = f"SELECT Plate_Number FROM database.csv WHERE RFID_Tag = '{RFID_Tag}'"
    result = duckdb.sql(query).fetchone()
    if result:
        plate_number = result[0]
        return plate_number

# Run the session (blocks until closed)
session.run()