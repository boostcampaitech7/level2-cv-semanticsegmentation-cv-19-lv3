import streamlit as st
import load, visualize

info = {
    'image_root': "../../data/train/DCM",
    'label_root': "../../data/train/outputs_json",
    'classes': [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ],
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    ],
    'color': {
        'finger': (120,203,228),
        'Trapezoid': (145,42,177),
        'Pisiform': (145,42,177),
        'Radius': (210,71,77),
        'Ulna': (210,71,77),
        'wrist': (193,223,159)
    }
}

st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Segmentation</h2>", unsafe_allow_html=True)

if "images" not in st.session_state:
    st.session_state.images = []
if "labels" not in st.session_state:
    st.session_state.labels = []

option = st.sidebar.radio("option", ["train", "validation", "test"])
if option == "train":
    st.session_state.images, st.session_state.labels = load.load(info)
    with st.sidebar.form(key="form"):
        st.session_state.text = st.selectbox("select text", ["None", "Text"])
        st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
        submit_button = st.form_submit_button("OK")
    image_count = st.sidebar.slider('Select image count', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-2, 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-(2*image_count), value=image_index, step=2*image_count)
    if image_index != image_index_input:
        image_index = image_index_input
    for i in range(image_index, image_index + 2 * image_count, 2):
        visualize.show(info, st.session_state.images, st.session_state.labels, i, st.session_state.text, st.session_state.anno)

if option == "validation":
    st.write("개발중입니다...")
if option == "test":
    st.write("개발중입니다...")