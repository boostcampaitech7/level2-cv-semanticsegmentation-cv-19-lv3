import os
import numpy as np
import pandas as pd
import streamlit as st
import load, visualize, visualize_rle, compare

def configure_sidebar(info, images, key_prefix="default"):
    with st.sidebar.form(key=f"{key_prefix}_anno_form"):
        anno = st.selectbox("Select annotation", ['All'] + info['classes'], key=f"{key_prefix}_anno")
        submit_button = st.form_submit_button("OK")
    
    image_count = st.sidebar.slider('Select image counts', 2, 8, 2, 2, key=f"{key_prefix}_count")
    image_line = st.sidebar.slider('Select image lines', 1, 4, 1, key=f"{key_prefix}_line")
    
    max_index = max(0, len(images) - image_count * image_line)
    image_index = st.sidebar.slider('Select image index', 0, max_index, key=f"{key_prefix}_index")
    image_index_input = st.sidebar.number_input(
        'Enter image index', 
        min_value=0, 
        max_value=max_index, 
        value=image_index,
        step=image_count * image_line, 
        key=f"{key_prefix}_index_input"
    )

    if image_index != image_index_input:
        image_index = image_index_input

    with st.sidebar.form(key="image name form"):
        image_name = st.text_input("Enter image name")
        submit_button = st.form_submit_button("OK")
        if submit_button and image_name:
            try:
                image_index = images.index(image_name)
                st.sidebar.success(f"Found at index {image_index}.")
            except IndexError:
                st.sidebar.error(f"Not Found")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")

    return anno, image_count, image_line, image_index

def main(info):
    st.sidebar.success("CV19 영원한종이박")
    st.markdown("<h2 style='text-align: center;'>Hand Bone Image Segmentation</h2>", unsafe_allow_html=True)

    option = st.sidebar.radio("option", ["Train Visualization", "Validation Compare", "Test Visualization", "RLE Visualization"])
    if option == "Train Visualization":
        st.session_state.images, st.session_state.labels = load.load(info, 'train')
        anno, image_count, image_line, image_index = configure_sidebar(info, st.session_state.images, key_prefix="train")
        for i in range(image_index, image_index + image_count * image_line, image_count):
            visualize.show(info, st.session_state.images, st.session_state.labels, [j for j in range(i, i + image_count, 2)], anno)

    if option == "Validation Compare":
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
            st.session_state.images = st.session_state.df['image_name'].unique().tolist()
            anno, image_count, image_line, image_index = configure_sidebar(info, st.session_state.images, key_prefix="valid")
            for i in range(image_index, image_index + image_line, 1):
                compare.show(info, st.session_state.df, st.session_state.images[i], anno)


    if option == "Test Visualization":
        st.session_state.images, _ = load.load(info, 'test')
        anno, image_count, image_line, image_index = configure_sidebar(info, st.session_state.images, key_prefix="test")
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
        anno, image_count, image_line, image_index = configure_sidebar(info, st.session_state.images, key_prefix="rle")
        for i in range(image_index, image_index + image_line*image_count, image_count):
            visualize_rle.show(info, st.session_state.df, st.session_state.images[i:i+image_count], anno)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    info = {
        'image_root': current_dir + "/../../data/{}/DCM",
        'label_root': current_dir + "/../../data/train/outputs_json",
        'classes': [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
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

    if "images" not in st.session_state:
        st.session_state.images = list()
    if "labels" not in st.session_state:
        st.session_state.labels = list()
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
        
    main(info)