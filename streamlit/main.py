import streamlit as st
import load, visualize_v1, visualize_v2, visualize_rle
import pandas as pd


#streamlit 실행하기 전에 del_ID_dir.py 실행 !!


info = {
    'image_root': "../../data/train/DCM",
    'all_image_root': "../../data/test/DCM/all_pngs",
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
st.markdown("<h2 style='text-align: center;'>Segmentation</h2>", unsafe_allow_html=True)

if "images" not in st.session_state:
    st.session_state.images = []
if "labels" not in st.session_state:
    st.session_state.labels = []

option = st.sidebar.radio("option", ["train", "validation", "test", "RLE Visualization"])
if option == "train":
    st.session_state.images, st.session_state.labels = load.load(info)
    with st.sidebar.form(key="form"):
        st.session_state.text = st.selectbox("select text", ["None", "Text"])
        st.session_state.anno = st.selectbox("select annotation", ['All'] + info['classes'])
        submit_button = st.form_submit_button("OK")
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.images)-2, 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.images)-2, value=image_index, step=2)
    if image_index != image_index_input:
        image_index = image_index_input
    visualize_v2.show(info, st.session_state.images, st.session_state.labels, image_index, st.session_state.text, st.session_state.anno)

if option == "validation":
    st.write("개발중입니다...")
if option == "test":
    st.write("개발중입니다...")
if option == "RLE Visualization":
    csv_file = st.sidebar.file_uploader("Upload RLE CSV file", type="csv")
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        df['rle'] = df['rle'].fillna('')  # NaN 값을 빈 문자열로 대체
        # 이미지 이름 목록 생성
        image_names = df['image_name'].unique()
        
        # 현재 이미지 인덱스를 세션 상태로 관리
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        
        # 이전 버튼
        if st.sidebar.button('Previous Image'):
            st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)
        
        # 다음 버튼
        if st.sidebar.button('Next Image'):
            st.session_state.current_image_index = min(len(image_names) - 1, st.session_state.current_image_index + 1)
        
        # 현재 이미지 이름 표시
        current_image = image_names[st.session_state.current_image_index]
        st.sidebar.write(f"Current Image: {current_image}")
        
        # 이미지 인덱스 슬라이더 (옵션)
        image_index = st.sidebar.slider('Select image index', 0, len(image_names)-1, st.session_state.current_image_index)
        st.session_state.current_image_index = image_index
        
        # 현재 선택된 이미지에 대한 데이터프레임 생성
        current_df = df[df['image_name'] == current_image]
        
        # 시각화 함수 호출
        visualize_rle.show(info, current_df, 0)