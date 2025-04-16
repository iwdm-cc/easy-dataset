import os
import glob
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

from TAIDI.llm.deepseek import get_openai_response

# 引入动态分块策略

# 思路
# 动态分块的思路：
# 内容分析：首先分析页面的结构，例如是否包含标题、段落、表格、图像等。不同的元素可能会对应不同的分块策略。
#
# 基于文本区域分块：如果页面包含多个段落或区域，基于文本行的高度或段落间距来划分块。
#
# 布局感知分块：基于图像中的布局，例如图像、表格、标题、正文等，动态地进行分块。
#
# 自适应调整：根据不同的页面内容和结构，调整分块的数量和大小。比如，如果某一块文本内容较少，可以适当增大分块的范围。

# 代码解释：
# dynamic_blocking 函数：
#
# 该函数通过计算每一行的“文本密度”来动态确定分块区域。假设文本行的密度较高时表示该区域有更多的内容，因此需要较小的块，而如果行的密度低，表示该区域是空白或边缘，可以适当合并或跳过。
#
# 通过设置一个min_block_height和max_block_height的范围，可以控制每个块的大小，避免块太小或太大。
#
# process_page_with_dynamic_blocks 函数：
#
# 调用dynamic_blocking来获取动态计算的块区域。
#
# 对每个块进行 OCR 提取文本。
#
# 优势：
# 动态调整：分块的高度和数量是动态决定的，不是固定的，可以根据每页的实际内容自动调整。
#
# 内容感知：能够更好地识别不同页面结构，如长文本段落和空白区域，可以避免过多的无效区域。
#
# 提高 OCR 精度：通过减少单个块中的无关信息（如空白区域），可能提升 OCR 识别精度。



INPUT_DIR = r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\datav2"  # 替换为实际输入目录路径
OUTPUT_DIR = r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\data7"  # 替换为实际输出目录路径，若不需要可设为 None
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
def convert_pdf_to_text_with_pymupdf(pdf_path):
    """使用 PyMuPDF 提取 PDF 中的文本"""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")  # 提取文本
        full_text += f"\n\nPage {page_number + 1}:\n{text}"

    return full_text


def dynamic_blocking(image, min_block_height=100, max_block_height=300):
    """
    动态分块处理，自动根据页面内容分割区域。
    假设内容分块是基于文本密度进行动态调整的。
    """
    width, height = image.size
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    # 计算每一行的“文本密度”
    text_density = np.sum(image_array, axis=1)  # 每行的像素强度和可以近似为文本密度

    # 基于文本密度的阈值来划分区域
    block_start = None
    blocks = []
    for i, density in enumerate(text_density):
        if density < 100 and block_start is not None:
            # 找到块的结束位置
            block_end = i
            block_height = block_end - block_start
            if min_block_height < block_height < max_block_height:
                blocks.append((0, block_start, width, block_end))  # x0, y0, x1, y1
            block_start = None
        elif density >= 100 and block_start is None:
            # 找到块的开始位置
            block_start = i

    # 如果最后一块未结束
    if block_start is not None:
        blocks.append((0, block_start, width, height))

    return blocks


def convert_pdf_to_text_with_ocr(pdf_path):
    """使用 pdf2image 和 pytesseract 提取 PDF 中图片的文本，并应用动态分块"""
    images = convert_from_path(pdf_path, dpi=300)  # 增加分辨率
    full_text = ""

    for page_number, image in enumerate(images):
        # 获取动态分块的区域
        blocks = dynamic_blocking(image)
        for box in blocks:
            block_image = image.crop(box)  # 获取分块图像
            text = pytesseract.image_to_string(block_image, lang='chi_sim')
            full_text += f"\n\nPage {page_number + 1} (Block):\n{text}"

    return full_text


def convert_pdf_to_text_with_pdfminer(pdf_path):
    """使用 pdfminer 提取 PDF 中的文本"""
    text = extract_text(pdf_path)
    return text


def fuse_texts(texts):
    """融合来自不同方法的文本"""
    # 使用TF-IDF和余弦相似度去重和融合
    vectorizer = TfidfVectorizer().fit_transform(texts)  # 将文本转换为TF-IDF向量
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer)  # 计算余弦相似度

    fused_text = texts[0]  # 默认保留第一个文本
    for idx, sim in enumerate(cosine_sim[0][1:], 1):  # 忽略与自身的相似度
        if sim < 0.85:  # 设置相似度阈值，若低于阈值则保留
            fused_text += f"\n\nMerged text from source {idx}:\n{texts[idx]}"

    return fused_text


def loadFile(param):
    """读取文件内容"""
    with open(param, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def optimize_text_with_model(fused_text):
    """使用预训练大模型优化融合的文本"""
    if not fused_text.strip():
        raise ValueError("输入文本为空或无效")

    prompt = loadFile(r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\llm\01_prompt.md")

    # 调用封装的大模型函数，获取优化后的文本
    optimized_text = get_openai_response(prompt, fused_text)

    return optimized_text

def optimize_text_with_model_2(fused_text):
    """使用预训练大模型优化融合的文本"""
    if not fused_text.strip():
        raise ValueError("输入文本为空或无效")

    prompt = loadFile(r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\llm\02_prompt.md")

    # 调用封装的大模型函数，获取优化后的文本
    optimized_text = get_openai_response(prompt, fused_text)

    return optimized_text

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：{INPUT_DIR} 不是一个目录")
        return

    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = glob.glob(os.path.join(INPUT_DIR, '*.pdf'))

    num = 0
    for pdf in pdf_files:
        print(num,"正在处理：", pdf)
        num += 1
        base, ext = os.path.splitext(pdf)
        output_txt = os.path.join(OUTPUT_DIR, os.path.basename(base) + '.txt')
        output_md = os.path.join(OUTPUT_DIR, os.path.basename(base) + '.md')

        try:
            # 提取多个来源的文本
            pymupdf_text = convert_pdf_to_text_with_pymupdf(pdf)
            ocr_text = convert_pdf_to_text_with_ocr(pdf)
            pdfminer_text = convert_pdf_to_text_with_pdfminer(pdf)

            # 融合多个方法的文本
            fused_text = fuse_texts([pymupdf_text, ocr_text, pdfminer_text])

            # 使用大模型进一步优化文本
            optimized_text = optimize_text_with_model(fused_text)
            optimized_text_2 = optimize_text_with_model_2(fused_text)
            print("优化后的文本：", optimized_text)

            # 保存优化后的文本
            with open(output_txt, 'w', encoding='utf-8') as output_file:
                output_file.write(optimized_text)

            with open(output_md, 'w', encoding='utf-8') as output_file:
                output_file.write(optimized_text_2)

            print(f"已处理 {pdf} 到 {output_txt} 和 {output_md}")

        except Exception as e:
            print(f"处理 {pdf} 时出错：{e}")


if __name__ == '__main__':
    main()
