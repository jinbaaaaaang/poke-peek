import json, torch, torch.nn as nn, torch.nn.functional as F
from io import BytesIO
from pathlib import Path
import requests, streamlit as st
from PIL import Image
from torchvision import models, transforms
try:
    import timm; HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]
IMAGE_SIZE=224
C1="#FFFEF9"; C2="#FFFDF5"; C3="#F0E8D5"; PINK="#F0C8CC"; DPINK="#C2185B"; TXT="#3D2B1F"; MTXT="#C0A070"

EXP_MAP={
    "Exp1: MobileNetV3-Small":("Exp1_MobileNetV3","mobilenet_v3_small"),
    "Exp2: ResNet-50":("Exp2_ResNet50","resnet50"),
    "Exp3: EfficientNet-B0":("Exp3_EfficientNetB0","efficientnet_b0"),
    "Exp4: ViT-B/16":("Exp4_ViT_B16","vit_b_16"),
}

def build_model(backbone,num_classes):
    if backbone=="mobilenet_v3_small":
        m=models.mobilenet_v3_small(weights=None)
        m.classifier[3]=nn.Linear(m.classifier[3].in_features,num_classes)
    elif backbone=="resnet50":
        m=models.resnet50(weights=None)
        m.fc=nn.Linear(m.fc.in_features,num_classes)
    elif backbone=="efficientnet_b0":
        m=models.efficientnet_b0(weights=None)
        m.classifier=nn.Sequential(nn.Dropout(0.2),nn.Linear(m.classifier[1].in_features,num_classes))
    elif backbone=="vit_b_16" and HAS_TIMM:
        m=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=num_classes)
    else: raise ValueError(f"Unknown: {backbone}")
    return m

@st.cache_resource
def load_model(exp_name,backbone,num_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=build_model(backbone,num_classes)
    m.load_state_dict(torch.load(f"results/checkpoints/{exp_name}.pth",map_location=device))
    return m.to(device).eval(),device

@st.cache_data
def load_classes():
    with open("results/class_names.json") as f: return json.load(f)

def preprocess(img):
    tf=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    return tf(img.convert("RGB")).unsqueeze(0)

import base64, io

def get_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

st.set_page_config(page_title="PokeClassifier",page_icon="🧁",layout="wide")
st.markdown(f"<style>[data-testid=stAppViewContainer]{{background:{C1}}}[data-testid=stSidebar]{{background:{C2}!important}}</style>",unsafe_allow_html=True)
st.markdown(f'<div style="background:{C2};border:1.5px solid {C3};padding:1.5rem;border-radius:20px;text-align:center;margin-bottom:1rem">'
            f'<div style="font-size:2rem">🧁</div>'
            f'<div style="color:{TXT};font-size:1.8rem;font-weight:800;margin:0">PokeClassifier</div>'
            f'<div style="color:{MTXT};font-size:.9rem;margin-top:.2rem">포켓몬 사진을 올리면 AI가 이름을 알아맞혀줘요!</div>'
            f'</div>',unsafe_allow_html=True)

class_names=load_classes()

with st.sidebar:
    st.markdown(f'<p style="color:{MTXT};font-weight:700">모델 선택</p>',unsafe_allow_html=True)
    choice=st.selectbox("",list(EXP_MAP.keys()),index=1)
    exp_name,backbone=EXP_MAP[choice]
    if backbone=="vit_b_16" and not HAS_TIMM: st.error("timm 필요!"); st.stop()
    try:
        with st.spinner("로딩 중..."):
            model,device=load_model(exp_name,backbone,len(class_names))
        st.success("준비 완료!")
        if Path("results/all_results.json").exists():
            with open("results/all_results.json") as f: res=json.load(f).get(exp_name,{})
            if res:
                st.markdown("---")
                st.markdown("**평가 결과**")
                c1,c2=st.columns(2)
                c1.metric("정확도",f"{res['test_accuracy']:.1%}")
                c2.metric("F1",f"{res['test_f1']:.1%}")
                c1.metric("정밀도",f"{res['test_precision']:.1%}")
                c2.metric("재현율",f"{res['test_recall']:.1%}")
    except FileNotFoundError: st.error("모델 없음! 학습 먼저!"); st.stop()

L,R=st.columns([1,1],gap="large")
with L:
    t1,t2=st.tabs(["사진 올리기","URL"])
    image=None
    with t1:
        up=st.file_uploader("",type=["jpg","jpeg","png","webp"])
        if up: image=Image.open(up).convert("RGB")
    with t2:
        url=st.text_input("",placeholder="https://...")
        if url:
            try: image=Image.open(BytesIO(requests.get(url,timeout=8).content)).convert("RGB")
            except: st.error("불러오기 실패")
    if image: 
        st.markdown(f'<div style="height:400px;display:flex;align-items:center;justify-content:center">'
                    f'<img src="data:image/png;base64,{get_base64(image)}" style="max-height:400px;max-width:100%;border-radius:12px;object-fit:contain">'
                    f'</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div style="text-align:center;padding:3rem;font-size:3rem">🧁</div>',unsafe_allow_html=True)

with R:
    if image is None:
        st.markdown(f'<div style="text-align:center;padding:4rem;color:{MTXT}"><div style="font-size:3rem">🧁</div><p>왼쪽에 이미지를 올려주세요!</p></div>',unsafe_allow_html=True)
    else:
        with st.spinner("분석 중..."):
            with torch.no_grad():
                probs=F.softmax(model(preprocess(image).to(device)),dim=1)[0]
        top_p,top_idx=torch.topk(probs,5)
        n1,c1val=class_names[top_idx[0]],top_p[0].item()
        st.markdown(f'<div style="background:{C2};border:2px solid {PINK};border-radius:16px;padding:1.2rem;text-align:center;margin-bottom:.8rem">'
                    f'<div style="font-size:1.5rem">🏆</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:{TXT}">{n1}</div>'
                    f'<div style="background:{PINK};display:inline-block;padding:.15rem .8rem;border-radius:99px;font-size:.85rem;color:{DPINK};font-weight:700;margin-top:.3rem">신뢰도 {c1val:.1%}</div>'
                    f'</div>',unsafe_allow_html=True)
        st.markdown("**다른 후보들**")
        medals=["🥈","🥉","4위","5위"]
        for i,(idx,p) in enumerate(zip(top_idx[1:],top_p[1:]),0):
            st.markdown(f"{medals[i]} **{class_names[idx]}** — {p.item():.1%}")
            st.progress(p.item())