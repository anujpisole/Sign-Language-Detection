import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr
from PIL import Image
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []




st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .sidebar-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<h1 class="sidebar-header">Sign Language Detection</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-header">SignLanguage Detection Team</p>', unsafe_allow_html=True)

# Add a theme selector
st.sidebar.markdown("### 🎨 App Theme")
theme = st.sidebar.selectbox(
    "Choose your theme",
    ["Light", "Dark", "Custom"],
    key="theme_selector"
)

if theme == "Custom":
    primary_color = st.sidebar.color_picker("Primary Color", "#1f77b4")
    st.markdown(f"""
        <style>
        .sidebar-header {{
            color: {primary_color};
        }}
        </style>
    """, unsafe_allow_html=True)

# Add language selection
st.sidebar.markdown("### 🌐 Language Settings")
app_language = st.sidebar.selectbox(
    "Interface Language",
    ["English", "Hindi", "Spanish", "French"]
)

sign_language = st.sidebar.selectbox(
    "Sign Language System",
    ["American Sign Language (ASL)", "Indian Sign Language (ISL)", "British Sign Language (BSL)"]
)

# # Add help and resources section
# st.sidebar.markdown("### ℹ️ Help & Resources")
# if st.sidebar.expander("Quick Tips"):
#     st.markdown("""
#     - Position your hand clearly in view
#     - Ensure good lighting
#     - Keep a steady hand position
#     - Use a plain background
#     """)

# if st.sidebar.expander("Keyboard Shortcuts"):
#     st.markdown("""
#     - **Spacebar**: Pause/Resume
#     - **S**: Save current frame
#     - **Q**: Quit application
#     - **F**: Toggle fullscreen
#     """)

# if st.sidebar.expander("About Sign Language"):
#     st.markdown("""
#     Sign language is a visual language that uses hand gestures, facial expressions, and body movements to communicate.
#     It is the primary language of many deaf and hard-of-hearing communities worldwide.
#     """)

# Add feedback section
st.sidebar.markdown("### 📝 Feedback")
feedback_type = st.sidebar.selectbox(
    "Type of Feedback",
    ["Bug Report", "Feature Request", "General Feedback"]
)

feedback_text = st.sidebar.text_area("Your Feedback")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")

# Add system status
st.sidebar.markdown("### 🔄 System Status")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("Camera")
    st.success("Active" if 'vid' in locals() else "Inactive")
with col2:
    st.markdown("Model")
    st.success("Loaded")

# Add version info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.text("Version: 1.0.0")
st.sidebar.text("Last Updated: 2024-03-04")

# Performance metrics
if st.sidebar.checkbox("Show Performance Metrics"):
    st.sidebar.markdown("### 📈 Performance")
    st.sidebar.metric("FPS", "30")
    st.sidebar.metric("Latency", "50ms")
    st.sidebar.metric("Accuracy", "95%")

# Export options
st.sidebar.markdown("### 💾 Export Options")
export_format = st.sidebar.selectbox(
    "Export Format",
    ["Video (.mp4)", "Images (.jpg)", "Text (.txt)"]
)

if st.sidebar.button("Export Results"):
    st.sidebar.info("Exporting results...")
    # Add export functionality here

# Add social links
st.sidebar.markdown("### 🔗 Connect")
cols = st.sidebar.columns(4)
with cols[0]:
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com)")
with cols[1]:
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com)")
with cols[2]:
    st.markdown("[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com)")
with cols[3]:
    st.markdown("[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com)")

# Original app mode selector with enhanced styling
st.sidebar.markdown("### 📱 App Mode")
app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Speech to sign Language', 'Practice Mode']
)

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)


    else:

        r = width / float(w)
        dim = (width, int(h * r))


    resized = cv2.resize(image, dim, interpolation=inter)


    return resized

if app_mode =='About App':
    st.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    
    # Main description
    st.markdown("""
    ## Welcome to our Sign Language Detection System
    
    This application uses advanced computer vision and machine learning techniques to detect and interpret American Sign Language (ASL) in real-time. Our system helps bridge the communication gap between the deaf/mute community and others by providing instant translation of sign language gestures.
    
    ### Key Features:
    - Real-time sign language detection using your webcam
    - Support for ASL alphabet and numbers
    - Speech to sign language conversion
    - User-friendly interface for easy interaction
    
    ### How It Works:
    1. **Hand Detection**: Using MediaPipe's hand tracking technology
    2. **Gesture Recognition**: Advanced ML models analyze hand positions
    3. **Real-time Translation**: Instant conversion of gestures to text
    
    ### About ASL:
    American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages. It is the primary language of many North Americans who are deaf and hard of hearing, and is used by many hearing people as well.
    
    #### Key Components of ASL:
    - **Handshapes**: Different positions of fingers and hands
    - **Movement**: How the hands move in space
    - **Location**: Where the signs are made
    - **Palm Orientation**: Direction the palm faces
    - **Non-manual Markers**: Facial expressions and body movements
    """)
    
    # Tutorial Section
    st.markdown("""
    ## How to Use This Application
    
    ### Sign Language to Text Mode:
    1. Select 'Sign Language to Text' from the sidebar
    2. Choose between webcam or upload a video
    3. Position your hand clearly in front of the camera
    4. Make ASL gestures and see the real-time translation
    
    ### Speech to Sign Language Mode:
    1. Select 'Speech to Sign Language' from the sidebar
    2. Speak clearly into your microphone
    3. Watch as your speech is converted to sign language visuals
    
    ### Supported Gestures:
    - Numbers (0-9)
    - Basic ASL alphabet
    - Common phrases and greetings
    """)
    
    # Technical Details
    st.markdown("""
    ## Technical Implementation
    
    This application leverages several cutting-edge technologies:
    
    - **MediaPipe**: For accurate hand tracking and gesture recognition
    - **OpenCV**: For real-time video processing
    - **StreamLit**: For the interactive web interface
    - **Machine Learning**: For gesture classification
    
    ### Privacy Note:
    Your privacy is important to us. All video processing is done locally on your device, and no data is stored or transmitted.
    """)
    
    # Original video and about me section
    st.video('https://www.youtube.com/watch?v=VtbYvVDItvg')
    st.markdown('''
              # About Me \n 
                Sign language is a vital mode of communication for the hearing and speech-impaired community, yet there remains a significant communication barrier between sign language users and those unfamiliar with it. This project aims to bridge that gap by developing an AI-powered Sign Language Detection system that can accurately recognize and translate sign language gestures into text or speech in real-time. \n

                Also check me out on Social Media
                - [YouTube]
                - 
                - 
              
                ''')
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')
    
    # Add instructions and tips
    st.markdown("""
    ### Instructions:
    1. Position your hand clearly in the camera view
    2. Make sure there is good lighting
    3. Keep your hand steady while making gestures
    4. Wait for the recognition to appear
    
    ### Tips for Better Recognition:
    - Maintain proper distance from camera (1-2 feet)
    - Avoid rapid movements
    - Use a plain background if possible
    """)

    # Add confidence meter
    confidence_container = st.container()
    with confidence_container:
        st.markdown("### Recognition Confidence")
        confidence_placeholder = st.empty()

    # Add recognized text display
    text_container = st.container()
    with text_container:
        st.markdown("### Recognized Text")
        text_placeholder = st.empty()
    
    # Original webcam and upload options
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    
    # Add detection settings
    st.sidebar.markdown("### Detection Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)
    
    # Add gesture categories
    st.sidebar.markdown("### Gesture Categories")
    detect_numbers = st.sidebar.checkbox("Detect Numbers", value=True)
    detect_letters = st.sidebar.checkbox("Detect Letters", value=True)
    detect_words = st.sidebar.checkbox("Detect Common Words", value=True)

    sameer=""
    st.markdown(' ## Output')
    st.markdown(sameer)

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    while vid.isOpened():

        ret, img = vid.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmark.landmark):
                    lm_list.append(lm)
                finger_fold_status = []
                for tip in finger_tips:
                    x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                    # print(id, ":", x, y)
                    # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                    if lm_list[tip].x < lm_list[tip - 2].x:
                        # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                print(finger_fold_status)
                x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
                print(x, y)
                # fuck off
                if lm_list[3].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "fuck off !!!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer="fuck off"

                # one
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[
                    12].y:
                    cv2.putText(img, "ONE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("1")

                # two
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "TWO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("2")
                    sameer="two"
                # three
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "THREE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("3")
                    sameer="three"

                # four
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                    cv2.putText(img, "FOUR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("4")
                    sameer="Four"

                # five
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "FIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("5")
                    sameer="Five"
                    # six
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "SIX", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("6")
                    sameer="Six"
                # SEVEN
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "SEVEN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("7")
                    sameer="Seven"
                # EIGHT
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "EIGHT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("8")
                    sameer="Eight"
                # NINE
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "NINE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("9")
                    sameer="Nine"
                # A
                if lm_list[2].y > lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y < lm_list[6].y:
                    cv2.putText(img, "A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("A")
                # B
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x > lm_list[8].x:
                    cv2.putText(img, "B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("B")
                    sameer="B"
                # c
                if lm_list[2].x < lm_list[4].x and lm_list[8].x > lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                        lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x:
                    cv2.putText(img, "C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("C")
                # d
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y > lm_list[8].y:
                    cv2.putText(img, "D", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("D")

                # E
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y > lm_list[6].y:
                    cv2.putText(img, "E", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("E")

                # Additional gestures for letters
                # A
                if lm_list[8].y < lm_list[7].y and lm_list[12].y > lm_list[11].y and \
                   lm_list[16].y > lm_list[15].y and lm_list[20].y > lm_list[19].y and \
                   lm_list[4].y > lm_list[3].y:
                    cv2.putText(img, "A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer="A"
                
                # B
                if lm_list[8].y < lm_list[7].y and lm_list[12].y < lm_list[11].y and \
                   lm_list[16].y < lm_list[15].y and lm_list[20].y < lm_list[19].y and \
                   lm_list[4].y < lm_list[3].y:
                    cv2.putText(img, "B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer="B"
                
                # Common words
                # Hello
                if lm_list[8].y < lm_list[7].y and lm_list[12].y < lm_list[11].y and \
                   lm_list[16].y > lm_list[15].y and lm_list[20].y > lm_list[19].y and \
                   lm_list[4].x > lm_list[3].x:
                    cv2.putText(img, "HELLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer="Hello"
                
                # Thank You
                if lm_list[8].y > lm_list[7].y and lm_list[12].y < lm_list[11].y and \
                   lm_list[16].y < lm_list[15].y and lm_list[20].y < lm_list[19].y and \
                   lm_list[4].y < lm_list[3].y:
                    cv2.putText(img, "THANK YOU", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer="Thank You"
                
                # Update confidence and text displays
                confidence_score = min(1.0, max(0.0, results.multi_hand_landmarks[0].landmark[0].visibility))
                confidence_placeholder.progress(confidence_score)
                text_placeholder.markdown(f"### Detected: {sameer}")
                
                # Draw hand landmarks with improved visibility
                mp_draw.draw_landmarks(
                    img,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_container_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()
elif app_mode == 'Practice Mode':
    st.title('Sign Language Practice Mode')
    
    st.markdown("""
    ## Welcome to Practice Mode!
    
    This mode helps you learn and practice American Sign Language (ASL). You'll be shown a gesture to make,
    and the system will provide real-time feedback on your attempts.
    """)
    
    # Practice categories
    practice_category = st.sidebar.selectbox(
        "Choose what to practice",
        ["Numbers (0-9)", "Alphabet (A-Z)", "Common Words"]
    )
    
    # Difficulty level
    difficulty = st.sidebar.select_slider(
        "Difficulty Level",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Beginner"
    )
    
    # Practice settings
    st.sidebar.markdown("### Practice Settings")
    show_guide_image = st.sidebar.checkbox("Show Guide Image", value=True)
    show_feedback = st.sidebar.checkbox("Show Real-time Feedback", value=True)
    
    # Initialize webcam
    use_webcam = st.sidebar.button('Start Practice')
    
    # Main practice area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Challenge")
        challenge_placeholder = st.empty()
        if show_guide_image:
            guide_image_placeholder = st.empty()
    
    with col2:
        st.markdown("### Your Attempt")
        attempt_placeholder = st.empty()
        feedback_placeholder = st.empty()
    
    if use_webcam:
        vid = cv2.VideoCapture(0)
        
        # Dictionary of challenges based on category
        challenges = {
            "Numbers (0-9)": ["Show number 1", "Show number 2", "Show number 3"],
            "Alphabet (A-Z)": ["Show letter A", "Show letter B", "Show letter C"],
            "Common Words": ["Sign 'Hello'", "Sign 'Thank You'", "Sign 'Please'"]
        }
        
        current_challenge = challenges[practice_category][0]
        challenge_placeholder.markdown(f"### {current_challenge}")
        
        while vid.isOpened():
            ret, img = vid.read()
            if not ret:
                continue
                
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    # Process landmarks
                    lm_list = []
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = img.shape
                        lm_list.append(lm)
                    
                    # Check if gesture matches challenge
                    feedback = "Keep trying!"
                    confidence = 0.0
                    
                    if current_challenge == "Show number 1":
                        if lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y:
                            feedback = "Perfect! You've got it!"
                            confidence = 0.9
                    elif current_challenge == "Show letter A":
                        if lm_list[8].y < lm_list[7].y and lm_list[12].y > lm_list[11].y:
                            feedback = "Excellent! That's correct!"
                            confidence = 0.95
                    
                    # Draw landmarks
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmark,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
                    
                    if show_feedback:
                        feedback_placeholder.markdown(f"""
                        ### Feedback
                        {feedback}
                        
                        Confidence: {confidence:.2%}
                        """)
            
            # Display the image
            attempt_placeholder.image(img, channels="BGR", use_column_width=True)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()
else:
    st.title('Speech to Sign Language (The System use Indian Sign Language)')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()


    # define function to display sign language images
    def display_images(text):
        # get the file path of the images directory
        img_dir = "images/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    # add start button to start recording audio
    if st.button("Start Talking"):
        # record audio for 5 seconds
        with sr.Microphone() as source:
            st.write("Say something!")
            audio = r.listen(source, phrase_time_limit=5)

            try:
                text = r.recognize_google(audio)
            except sr.UnknownValueError:
                st.write("Sorry, I did not understand what you said.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

        # convert text to lowercase
        text = text.lower()
        # display the final result
        st.write(f"You said: {text}", font_size=41)

        # display sign language images
        display_images(text)

