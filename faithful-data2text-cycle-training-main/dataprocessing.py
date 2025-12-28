# import os
# import xml.etree.ElementTree as ET
#
# # === 输入与输出路径 ===
# root_dir = r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\datasets\webnlg-dataset-master\webnlg-dataset-master\release_v3.0\en"
# output_base = r"G:\abnormal\cycletext\data"
#
# splits = ["train", "dev", "test"]
#
# # faithful-data2text-cycle-training 所需输出路径
# out_trip2text = os.path.join(output_base, "webnlg-t5-triplets2text")
# out_text2trip = os.path.join(output_base, "webnlg-t5-text2triplets")
# os.makedirs(out_trip2text, exist_ok=True)
# os.makedirs(out_text2trip, exist_ok=True)
#
#
# def extract_entries(xml_path, is_test=False):
#     """从 WebNLG v3.0 XML 文件中提取 (triples, text) 对"""
#     pairs = []
#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#
#         for entry in root.iter("entry"):
#             triples = []
#             # 支持 modifiedtripleset, originaltripleset, tripleset
#             for node_name in ["modifiedtripleset", "originaltripleset", "tripleset"]:
#                 for tag in ["mtriple", "otriple", "triple"]:
#                     for triple in entry.findall(f".//{node_name}/{tag}"):
#                         if triple.text:
#                             triples.append(triple.text.strip())
#
#             if not triples:
#                 continue
#             triple_str = " && ".join(triples)
#
#             # 提取自然语言句子（新版是 <lex>）
#             texts = []
#             for lex in entry.findall(".//lex"):
#                 if lex.text:
#                     text = lex.text.strip()
#                     if text:
#                         texts.append(text)
#
#             if texts:
#                 for text in texts:
#                     pairs.append((triple_str, text))
#             else:
#                 # 测试集没有文本时保留 triple，text 设为 None
#                 if is_test:
#                     pairs.append((triple_str, None))
#     except Exception as e:
#         print(f"[Error parsing] {xml_path}: {e}")
#     return pairs
#
#
# for split in splits:
#     print(f"\n📂 Processing {split}...")
#     split_dir = os.path.join(root_dir, split)
#     all_pairs = []
#     is_test = split == "test"
#
#     # === 遍历所有子文件夹 (1triples, 2triples...) ===
#     for subdir, _, files in os.walk(split_dir):
#         for fname in files:
#             if fname.endswith(".xml"):
#                 xml_path = os.path.join(subdir, fname)
#                 pairs = extract_entries(xml_path, is_test=is_test)
#                 if len(pairs) == 0:
#                     print(f"⚠️ no pairs extracted from {xml_path}")
#                 else:
#                     print(f"✅ {len(pairs)} pairs extracted from {os.path.relpath(xml_path, split_dir)}")
#                 all_pairs.extend(pairs)
#
#     print(f"→ Total extracted {len(all_pairs)} pairs for {split}")
#
#     # === 输出 triplets2text (三元组→文本) ===
#     data2text_source = [f"Generate in English: {t}" for t, _ in all_pairs]
#     data2text_tsv = [f"Generate in English: {t}\t{s}" for t, s in all_pairs if s is not None]
#
#     split_name = "train" if split == "train" else "val" if split == "dev" else "test"
#
#     with open(os.path.join(out_trip2text, f"{split_name}.source"), "w", encoding="utf-8") as f:
#         f.write("\n".join(data2text_source))
#     with open(os.path.join(out_trip2text, f"{split_name}.tsv"), "w", encoding="utf-8") as f:
#         f.write("\n".join(data2text_tsv))
#
#     # === 输出 text2triplets (文本→三元组，仅 train/dev) ===
#     if not is_test:
#         text2data_source = [f"Extract Triplets: {s}" for _, s in all_pairs]
#         text2data_tsv = [f"Extract Triplets: {s}\t{t}" for t, s in all_pairs]
#
#         with open(os.path.join(out_text2trip, f"{split_name}.source"), "w", encoding="utf-8") as f:
#             f.write("\n".join(text2data_source))
#         with open(os.path.join(out_text2trip, f"{split_name}.tsv"), "w", encoding="utf-8") as f:
#             f.write("\n".join(text2data_tsv))
#
# print("\n✅ 数据集提取与格式转换完成，可直接用于 faithful-data2text-cycle-training 项目！")
# import os
# import random
# import glob
# import xml.etree.ElementTree as ET
# from tqdm import tqdm
#
#
# def extract_triples_from_xml(xml_path):
#     """从 WebNLG XML 文件中提取 (triples, text) 对"""
#     pairs = []
#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#     except Exception as e:
#         print(f"⚠️ 无法解析 {xml_path}: {e}")
#         return pairs
#
#     # 寻找所有 entry 节点（兼容不同版本）
#     entries = root.findall(".//entry")
#     if not entries:
#         entries = root.findall(".//benchmark/entries/entry")
#
#     for entry in entries:
#         triples = []
#         for mtriple in entry.findall(".//modifiedtripleset/mtriple"):
#             if mtriple.text:
#                 triples.append(mtriple.text.strip())
#         triple_str = " | ".join(triples)
#         if not triple_str:
#             continue
#
#         for lex in entry.findall(".//lex"):
#             if lex.text:
#                 text = lex.text.strip().replace("\n", " ")
#                 pairs.append((triple_str, text))
#
#     return pairs
#
#
# def process_webnlg_dataset(base_dir, output_dir):
#     """主处理逻辑：生成 full、100-paired、unpaired 三套数据"""
#     splits = ["train", "dev", "test"]
#     all_pairs = []
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     print("🔍 开始解析 WebNLG 数据集...")
#     for split in splits:
#         # WebNLG v3.0 实际目录：release_v3.0/en/train/1triples/*.xml
#         xml_dir = os.path.join(base_dir, "en", split)
#         print(f"\n📂 当前处理路径: {xml_dir}")
#
#         xml_files = glob.glob(os.path.join(xml_dir, "**/*.xml"), recursive=True)
#         print(f"共找到 {len(xml_files)} 个 XML 文件。")
#
#         split_pairs = []
#         for xml_file in tqdm(xml_files, desc=f"Processing {split}"):
#             split_pairs.extend(extract_triples_from_xml(xml_file))
#
#         # 保存完整paired数据
#         split_dir_data2text = os.path.join(output_dir, f"webnlg-t5-triplets2text/{split}")
#         split_dir_text2data = os.path.join(output_dir, f"webnlg-t5-text2triplets/{split}")
#         os.makedirs(split_dir_data2text, exist_ok=True)
#         os.makedirs(split_dir_text2data, exist_ok=True)
#
#         data2text_path = os.path.join(split_dir_data2text, f"{split}.tsv")
#         text2data_path = os.path.join(split_dir_text2data, f"{split}.tsv")
#
#         with open(data2text_path, "w", encoding="utf-8") as f1, open(text2data_path, "w", encoding="utf-8") as f2:
#             for triple, text in split_pairs:
#                 f1.write(f"Generate in English: {triple}\t{text}\n")
#                 f2.write(f"Extract Triplets: {text}\t{triple}\n")
#
#         print(f"✅ {split} 完整数据保存完成，共 {len(split_pairs)} 对样本。")
#
#         if split == "train":
#             all_pairs.extend(split_pairs)
#
#     # === 构建低资源版本 ===
#     print("\n🎯 开始构建低资源版本（100条 + unpaired）...")
#     random.seed(42)
#     random.shuffle(all_pairs)
#
#     paired_100 = all_pairs[:100]
#     unpaired_rest = all_pairs[100:]
#
#     # 100条 paired
#     paired_dir = os.path.join(output_dir, "webnlg-t5-100paired")
#     os.makedirs(paired_dir, exist_ok=True)
#     with open(os.path.join(paired_dir, "train_data2text.tsv"), "w", encoding="utf-8") as f:
#         for t, s in paired_100:
#             f.write(f"Generate in English: {t}\t{s}\n")
#     with open(os.path.join(paired_dir, "train_text2data.tsv"), "w", encoding="utf-8") as f:
#         for t, s in paired_100:
#             f.write(f"Extract Triplets: {s}\t{t}\n")
#     print(f"✅ 已生成 100 条 paired 数据。")
#
#     # 剩余 unpaired
#     unpaired_dir = os.path.join(output_dir, "webnlg-t5-unpaired")
#     os.makedirs(unpaired_dir, exist_ok=True)
#     triples_unpaired = [f"Generate in English: {t}" for t, _ in unpaired_rest]
#     texts_unpaired = [f"Extract Triplets: {s}" for _, s in unpaired_rest]
#
#     with open(os.path.join(unpaired_dir, "triples_unpaired.txt"), "w", encoding="utf-8") as f:
#         f.write("\n".join(triples_unpaired))
#     with open(os.path.join(unpaired_dir, "texts_unpaired.txt"), "w", encoding="utf-8") as f:
#         f.write("\n".join(texts_unpaired))
#
#     print(f"✅ 已生成 unpaired 数据：{len(unpaired_rest)} 条。")
#
#     print("\n🎉 数据构建完成！输出结构如下：")
#     print(f"{output_dir}/")
#     print("├── webnlg-t5-triplets2text/ (完整训练集/验证集/测试集)")
#     print("├── webnlg-t5-text2triplets/ (完整训练集/验证集/测试集)")
#     print("├── webnlg-t5-100paired/ (100条有标注样本)")
#     print("└── webnlg-t5-unpaired/ (剩余无标注数据)")
#     print("✅ 可直接用于基线模型和 CycleText 模型训练。")
#
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Convert WebNLG XML to CycleText format")
#     parser.add_argument("--webnlg_dir", type=str, required=True,
#                         help="WebNLG 数据集的根目录（包含 release_v3.0/en/train/...）")
#     parser.add_argument("--output_dir", type=str, required=True,
#                         help="输出路径（将生成多个子文件夹）")
#     args = parser.parse_args()
#
#     process_webnlg_dataset(args.webnlg_dir, args.output_dir)
import os
import random
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm


def extract_triples_from_xml(xml_path):
    """从 WebNLG XML 文件中提取 (triples, text) 对"""
    pairs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"⚠️ 无法解析 {xml_path}: {e}")
        return pairs

    entries = root.findall(".//entry")
    if not entries:
        entries = root.findall(".//benchmark/entries/entry")

    for entry in entries:
        triples = []
        for mtriple in entry.findall(".//modifiedtripleset/mtriple"):
            if mtriple.text:
                triples.append(mtriple.text.strip())
        triple_str = " | ".join(triples)
        if not triple_str:
            continue

        for lex in entry.findall(".//lex"):
            if lex.text:
                text = lex.text.strip().replace("\n", " ")
                pairs.append((triple_str, text))

    return pairs


def process_webnlg_dataset(base_dir, output_dir):
    """生成 full、100-paired、unpaired 三套数据 (含 .tsv, .source, .target)"""
    splits = ["train", "dev", "test"]
    all_pairs = []

    os.makedirs(output_dir, exist_ok=True)

    print("🔍 开始解析 WebNLG 数据集...")
    for split in splits:
        xml_dir = os.path.join(base_dir, "en", split)
        print(f"\n📂 当前处理路径: {xml_dir}")

        xml_files = glob.glob(os.path.join(xml_dir, "**/*.xml"), recursive=True)
        print(f"共找到 {len(xml_files)} 个 XML 文件。")

        split_pairs = []
        for xml_file in tqdm(xml_files, desc=f"Processing {split}"):
            split_pairs.extend(extract_triples_from_xml(xml_file))

        # === 保存完整 paired 数据 ===
        split_dir_data2text = os.path.join(output_dir, f"webnlg-t5-triplets2text/{split}")
        split_dir_text2data = os.path.join(output_dir, f"webnlg-t5-text2triplets/{split}")
        os.makedirs(split_dir_data2text, exist_ok=True)
        os.makedirs(split_dir_text2data, exist_ok=True)

        data2text_tsv = os.path.join(split_dir_data2text, f"{split}.tsv")
        text2data_tsv = os.path.join(split_dir_text2data, f"{split}.tsv")
        data2text_source = os.path.join(split_dir_data2text, f"{split}.source")
        data2text_target = os.path.join(split_dir_data2text, f"{split}.target")
        text2data_source = os.path.join(split_dir_text2data, f"{split}.source")
        text2data_target = os.path.join(split_dir_text2data, f"{split}.target")

        with open(data2text_tsv, "w", encoding="utf-8") as f1, \
             open(text2data_tsv, "w", encoding="utf-8") as f2, \
             open(data2text_source, "w", encoding="utf-8") as s1, \
             open(data2text_target, "w", encoding="utf-8") as t1, \
             open(text2data_source, "w", encoding="utf-8") as s2, \
             open(text2data_target, "w", encoding="utf-8") as t2:

            for triple, text in split_pairs:
                # 方向1: triples -> text
                f1.write(f"Generate in English: {triple}\t{text}\n")
                s1.write(f"Generate in English: {triple}\n")
                t1.write(f"{text}\n")
                # 方向2: text -> triples
                f2.write(f"Extract Triplets: {text}\t{triple}\n")
                s2.write(f"Extract Triplets: {text}\n")
                t2.write(f"{triple}\n")

        print(f"✅ {split} 完整数据保存完成，共 {len(split_pairs)} 对样本。")

        if split == "train":
            all_pairs.extend(split_pairs)

    # === 构建低资源版本 ===
    print("\n🎯 开始构建低资源版本（100条 + unpaired）...")
    random.seed(42)
    random.shuffle(all_pairs)

    paired_100 = all_pairs[:100]
    unpaired_rest = all_pairs[100:]

    # 100 条 paired
    paired_dir = os.path.join(output_dir, "webnlg-t5-100paired")
    os.makedirs(paired_dir, exist_ok=True)
    with open(os.path.join(paired_dir, "train_data2text.tsv"), "w", encoding="utf-8") as f1, \
         open(os.path.join(paired_dir, "train_text2data.tsv"), "w", encoding="utf-8") as f2, \
         open(os.path.join(paired_dir, "train_data2text.source"), "w", encoding="utf-8") as s1, \
         open(os.path.join(paired_dir, "train_data2text.target"), "w", encoding="utf-8") as t1, \
         open(os.path.join(paired_dir, "train_text2data.source"), "w", encoding="utf-8") as s2, \
         open(os.path.join(paired_dir, "train_text2data.target"), "w", encoding="utf-8") as t2:

        for t, s in paired_100:
            f1.write(f"Generate in English: {t}\t{s}\n")
            f2.write(f"Extract Triplets: {s}\t{t}\n")
            s1.write(f"Generate in English: {t}\n")
            t1.write(f"{s}\n")
            s2.write(f"Extract Triplets: {s}\n")
            t2.write(f"{t}\n")

    print(f"✅ 已生成 100 条 paired 数据。")

    # 剩余 unpaired
    unpaired_dir = os.path.join(output_dir, "webnlg-t5-unpaired")
    os.makedirs(unpaired_dir, exist_ok=True)
    triples_unpaired = [f"Generate in English: {t}" for t, _ in unpaired_rest]
    texts_unpaired = [f"Extract Triplets: {s}" for _, s in unpaired_rest]

    with open(os.path.join(unpaired_dir, "triples_unpaired.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(triples_unpaired))
    with open(os.path.join(unpaired_dir, "texts_unpaired.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(texts_unpaired))

    print(f"✅ 已生成 unpaired 数据：{len(unpaired_rest)} 条。")

    print("\n🎉 数据构建完成！输出结构如下：")
    print(f"{output_dir}/")
    print("├── webnlg-t5-triplets2text/ (完整训练集/验证集/测试集, 含 .source/.target)")
    print("├── webnlg-t5-text2triplets/ (完整训练集/验证集/测试集, 含 .source/.target)")
    print("├── webnlg-t5-100paired/ (100条有标注样本, 含 .source/.target)")
    print("└── webnlg-t5-unpaired/ (剩余无标注数据)")
    print("✅ 可直接用于基线模型和 CycleText 模型训练。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert WebNLG XML to CycleText format (.source/.target included)")
    parser.add_argument("--webnlg_dir", type=str, required=True,
                        help="WebNLG 数据集根目录（包含 release_v3.0/en/train/...）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出路径（将生成多个子文件夹）")
    args = parser.parse_args()

    process_webnlg_dataset(args.webnlg_dir, args.output_dir)

