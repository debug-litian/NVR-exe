import numpy as np
import torch
import clip
import faiss
import pickle

# ==================== 配置项（和上面保持一致）====================
INDEX_PATH = "./video_index.faiss"
MAPPING_PATH = "./frame_mapping.pkl"
TOP_K = 10        # 返回前10条结果
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================================

# 加载模型、索引、映射表
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()
index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    frame_mapping = pickle.load(f)

def search_by_text(text_query):
    # 文本编码
    text_token = clip.tokenize([text_query]).to(DEVICE)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    text_feat_np = text_feat.cpu().numpy().astype(np.float32)

    # FAISS检索
    distances, indices = index.search(text_feat_np, TOP_K)
    print(f"\n===== 检索关键词：{text_query} =====")
    print(f"排名\t视频文件\t时间戳(秒)\t相似度距离")
    for rank, idx in enumerate(indices[0]):
        video_name, ts = frame_mapping[idx]
        dist = round(distances[0][rank], 4)
        print(f"{rank+1}\t{video_name}\t{ts}s\t\t{dist}")

if __name__ == "__main__":
    print("文搜视频检索工具（输入 quit 退出）")
    while True:
        query = input("\n请输入检索文本：")
        if query.strip().lower() == "quit":
            break
        if not query.strip():
            continue
        search_by_text(query)