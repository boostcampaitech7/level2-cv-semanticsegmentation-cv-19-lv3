import streamlit as st
import load, visualize, visualize_rle, compare
import numpy as np
import pandas as pd


#streamlit 실행하기 전에 del_ID_dir.py 실행 !!


info = {
    'image_root': "/data/ephemeral/home/sy/level2-cv-semanticsegmentation-cv-19-lv3/data/test/DCM",
    'label_root': "/data/ephemeral/home/sy/level2-cv-semanticsegmentation-cv-19-lv3/data/train/outputs_json",
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
        'Trapezium': (0,85,0),
        'Capitate': (230,114,61),
        'Hamate': (196,183,59),
        'Scaphoid': (120,240,50),
        'Lunate': (107,102,255),
        'Triquetrum': (116,116,116),
        'wrist': (193,223,159)
    }
}

st.sidebar.success("CV19 영원한종이박")
st.markdown("<h2 style='text-align: center;'>Hand Bone Image Segmentation</h2>", unsafe_allow_html=True)

if "images" not in st.session_state:
    st.session_state.images = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

option = st.sidebar.radio("option", ["train Visualization", "validation Compare", "test Visualization", "RLE Visualization"])
if option == "train Visualization":
    st.session_state.images, st.session_state.labels = load.load(info, 'train')
    with st.sidebar.form(key="form"):
        st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
        submit_button = st.form_submit_button("OK")
    image_count = st.sidebar.slider('Select image count', 2, 8, 2, 2)
    image_line = st.sidebar.slider('Select image line', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-(image_count*image_line), 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-(image_count*image_line), value=image_index, step=image_count*image_line)
    if image_index != image_index_input:
        image_index = image_index_input
    for i in range(image_index, image_index + image_count * image_line, image_count):
        visualize.show(info, st.session_state.images, st.session_state.labels, [j for j in range(i, i + image_count, 2)], st.session_state.anno)

if option == "validation Compare":
    with st.sidebar.form(key="csv form"):
        csv_path = st.text_input("csv file path")
        submit_button = st.form_submit_button("OK")
    if submit_button and csv_path:
        try:
            st.session_state.df = load.load_csv(info, csv_path, 'train')
            st.sidebar.success("csv file load successed :)")
        except Exception:
            st.sidebar.error("csv file load failed :(")
    if st.session_state.df.empty:
        st.stop()
    if not st.session_state.df.empty:
        with st.sidebar.form(key="detail form"):
            st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
            submit_button = st.form_submit_button("OK")
        st.session_state.images = st.session_state.df['image_name'].unique().tolist()
        image_line = st.sidebar.slider('Select image line', 1, 4, 1)
        image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-image_line, 0)
        image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-image_line, value=image_index, step=image_line)
        if image_index != image_index_input:
            image_index = image_index_input
        for i in range(image_index, image_index + image_line, 1):
            compare.show(info, st.session_state.df, st.session_state.images[i], st.session_state.anno)


if option == "test Visualization":
    st.session_state.images, _ = load.load(info, 'test')
    image_count = st.sidebar.slider('Select image count', 2, 8, 2, 2)
    image_line = st.sidebar.slider('Select image line', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-(image_count*image_line), 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-(image_count*image_line), value=image_index, step=image_count*image_line)
    if image_index != image_index_input:
        image_index = image_index_input
    for i in range(image_index, image_index + image_count * image_line, image_count):
        visualize.show(info, st.session_state.images, np.array([]), [j for j in range(i, i + image_count, 2)], '')

if option == "RLE Visualization":
    csv_file = st.sidebar.file_uploader("Upload RLE CSV file", type="csv")
    with st.sidebar.form(key="csv form"):
        csv_path = st.text_input("csv file path")
        submit_button = st.form_submit_button("OK")
    if submit_button and csv_path:
        try:
            st.session_state.df = load.load_csv(info, csv_path, 'test')
            st.sidebar.success("csv file load successed :)")
        except Exception:
            st.sidebar.error("csv file load failed :(")
    if csv_file is not None:
        st.session_state.df = load.load_csv(info, csv_file, 'test')
        st.sidebar.success("csv file load successed :)")
    if st.session_state.df.empty:
        st.stop()
    if not st.session_state.df.empty:
        with st.sidebar.form(key="detail form"):
            st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
            submit_button = st.form_submit_button("OK")
    st.session_state.images = st.session_state.df['image_name'].unique().tolist()
    image_count = st.sidebar.slider('Select image count', 1, 4, 1)
    image_line = st.sidebar.slider('Select image line', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-image_count*image_line, 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-image_count*image_line, value=image_index, step=image_count*image_line)
    if image_index != image_index_input:
        image_index = image_index_input
    for i in range(image_index, image_index + image_line*image_count, image_count):
        visualize_rle.show(info, st.session_state.df, st.session_state.images[i:i+image_count], st.session_state.anno)