import os
import numpy as np
import pandas as pd
import load
import compare
import visualize
import visualize_rle
import streamlit as st

def configure_sidebar(info: dict, images: list, key_prefix: str="default") -> tuple:
    with st.sidebar.form(key=f"{key_prefix}_anno_form"):
        anno = st.selectbox(
            label="Select annotation",
            options=['All']+info['classes'],
            key=f"{key_prefix}_anno"
        )
        submit_button = st.form_submit_button("OK")
    count = st.sidebar.slider('Select image counts', 2, 8, 2, 2, key=f"{key_prefix}_count")
    line = st.sidebar.slider('Select image lines', 1, 4, 1, key=f"{key_prefix}_line")
    max_index = max(0, len(images) - count * line)
    index = st.sidebar.slider('Select image index', 0, max_index, key=f"{key_prefix}_index")
    index_input = st.sidebar.number_input(
        'Enter image index', 
        min_value=0,
        max_value=max_index,
        value=index,
        step=count * line,
        key=f"{key_prefix}_index_input"
    )

    if index != index_input:
        index = index_input

    with st.sidebar.form(key="image name form"):
        image_name = st.text_input("Enter image name")
        submit_button = st.form_submit_button("OK")
        if submit_button and image_name:
            try:
                index = images.index(image_name)
                st.sidebar.success(f"Found at index {index}")
            except Exception:
                st.sidebar.error(f"Not Found '{image_name}'")

    return anno, count, line, index

def main(info: dict) -> None:
    st.sidebar.success(body="CV19 Forever Paper Box")
    st.markdown(
        body="<h2 style='text-align: center;'>Hand Bone Image Segmentation</h2>",
        unsafe_allow_html=True
    )

    option = st.sidebar.radio(
        label="option",
        options=[
            "Train Visualization",
            "Validation Compare",
            "Test Visualization",
            "RLE Visualization"
        ]
    )

    if option == "Train Visualization":
        st.session_state.images, st.session_state.labels = load.load(info, 'train')
        anno, count, line, index = configure_sidebar(
            info=info,
            images=st.session_state.images,
            key_prefix="train"
        )
        for i in range(index, index + count * line, count):
            visualize.show(
                info=info,
                images=st.session_state.images,
                labels=st.session_state.labels,
                ids=list(range(i, i + count, 2)),
                anno=anno
            )

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
            anno, count, line, index = configure_sidebar(
                info=info,
                images=st.session_state.images,
                key_prefix="valid"
            )
            for i in range(index, index+line, 1):
                compare.show(
                    info=info,
                    df=st.session_state.df,
                    image_name=st.session_state.images[i],
                    anno=anno
                )

    if option == "Test Visualization":
        st.session_state.images, _ = load.load(info, 'test')
        anno, count, line, index = configure_sidebar(
            info=info,
            images=st.session_state.images,
            key_prefix="test"
        )
        for i in range(index, index+count*line, count):
            visualize.show(
                info=info,
                images=st.session_state.images,
                labels=np.array([]),
                ids=list(range(i, i + count, 2)),
                anno=''
            )

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
        elif not st.session_state.df.empty:
            with st.sidebar.form(key="detail form"):
                st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
                submit_button = st.form_submit_button("OK")
        st.session_state.images = st.session_state.df['image_name'].unique().tolist()
        anno, count, line, index = configure_sidebar(
            info=info,
            images=st.session_state.images,
            key_prefix="rle"
        )
        for i in range(index, index + line*count, count):
            visualize_rle.show(
                info=info,
                df=st.session_state.df,
                images=st.session_state.images[i:i+count],
                anno=anno
            )

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
        st.session_state.images = []
    if "labels" not in st.session_state:
        st.session_state.labels = []
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    main(info)
