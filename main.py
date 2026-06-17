import cv2
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

# Configure streaming options
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

# Run the session (blocks until closed)
session.run()