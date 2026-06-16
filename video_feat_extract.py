import os
import cv2
import numpy as np
import torch
import clip
import faiss
import pickle

# ==================== 配置项（自己改这里）====================
VIDEO_DIR = "./nvr_video"  # 你的视频文件所在目录
INDEX_PATH = "./video_index.faiss"  # 特征索引文件
INDEX_SAVE_PATH = "./video_index.faiss"  # faiss特征索引文件，存视频帧的向量特征，读写共用
MAPPING_PATH = "./frame_mapping.pkl"   # 帧-时间戳映射文件
MAPPING_SAVE_PATH = "./frame_mapping.pkl"  # 映射文件，记录「特征向量 → 原视频+时间戳」对应关系
TOP_K = 10          # 检索时，最多返回10条匹配结果  
SAMPLE_FPS = 1  # 每秒抽1帧，可根据需要调整
DEVICE = "cpu"  # 强制使用CPU运行python video_feat_extract.py模型
# ===========================================================

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def extract_frames(video_path, sample_fps):
    """按帧率抽帧，返回(帧图像, 对应时间戳)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / sample_fps)
    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # BGR转RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            # 记录当前帧时间(秒)
            ts = frame_idx / fps
            timestamps.append(round(ts, 2))
        frame_idx += 1
    cap.release()
    return frames, timestamps

def main():
    all_features = []
    all_mapping = []  # 每条：(视频文件名, 帧时间戳)

    # 遍历文件夹下所有mp4视频
    for video_name in os.listdir(VIDEO_DIR):
        if not video_name.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(VIDEO_DIR, video_name)
        print(f"正在处理：{video_name}")

        frames, timestamps = extract_frames(video_path, SAMPLE_FPS)
        if len(frames) == 0:
            print(f"警告：{video_name} 无有效帧，跳过")
            continue

        # CLIP预处理 + 编码
        from PIL import Image
        img_input = torch.stack([preprocess(Image.fromarray(frame[...,::-1])) for frame in frames]).to(DEVICE)
        with torch.no_grad():
            feat = model.encode_image(img_input)
        # 向量归一化（检索核心，必须加）
        feat /= feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy().astype(np.float32)

        all_features.append(feat_np)
        # 记录映射关系
        for ts in timestamps:
            all_mapping.append((video_name, ts))

    # 拼接所有向量
    total_feats = np.vstack(all_features)
    dim = total_feats.shape[1]
    print(f"总向量数：{total_feats.shape[0]}, 向量维度：{dim}")

    # 构建FAISS索引（CPU通用版）
    index = faiss.IndexFlatL2(dim)
    index.add(total_feats)

    # 保存索引和映射表
    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(MAPPING_SAVE_PATH, "wb") as f:
        pickle.dump(all_mapping, f)

    print("特征库 & 映射表 保存完成！")

if __name__ == "__main__":
    main()