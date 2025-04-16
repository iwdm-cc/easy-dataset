import glob
import os

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# 定义输入和输出目录路径
INPUT_DIR = r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\data"  # 替换为实际输入目录路径
OUTPUT_DIR = r"C:\Users\47306\PycharmProjects\easy-dataset\TAIDI\data5"  # 替换为实际输出目录路径，若不需要可设为 None


def convert_pdf_to_text_with_pymupdf(pdf_path):
    """使用 PyMuPDF 提取 PDF 中的文本"""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")  # 提取文本
        full_text += f"\n\nPage {page_number + 1}:\n{text}"

    return full_text


def convert_pdf_to_text_with_ocr(pdf_path):
    """使用 pdf2image 和 pytesseract 提取 PDF 中图片的文本"""
    images = convert_from_path(pdf_path, dpi=300)  # 增加分辨率
    full_text = ""
    for page_number, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='chi_sim')
        full_text += f"\n\nPage {page_number + 1}:\n{text}"

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


def optimize_text_with_model(fused_text):
    """使用预训练大模型优化融合的文本"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    optimized_text = summarizer(fused_text, max_length=500, min_length=50, do_sample=False)
    return optimized_text[0]['summary_text']


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：{INPUT_DIR} 不是一个目录")
        return

    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = glob.glob(os.path.join(INPUT_DIR, '*.pdf'))

    for pdf in pdf_files:
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

            # 保存优化后的文本
            with open(output_txt, 'w', encoding='utf-8') as output_file:
                output_file.write(optimized_text)

            with open(output_md, 'w', encoding='utf-8') as output_file:
                output_file.write(optimized_text)

            print(f"已处理 {pdf} 到 {output_txt} 和 {output_md}")

        except Exception as e:
            print(f"处理 {pdf} 时出错：{e}")


if __name__ == '__main__':
    main()
