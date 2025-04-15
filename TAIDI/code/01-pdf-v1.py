import os
import glob
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance

# 定义输入和输出目录路径
INPUT_DIR = r"C:\Users\admin\PycharmProjects\easy-dataset\TAIDI\data"  # 替换为实际输入目录路径
OUTPUT_DIR = r"C:\Users\admin\PycharmProjects\easy-dataset\TAIDI\data1"  # 替换为实际输出目录路径，若不需要可设为 None

def preprocess_image(image):
    """图像预处理，包括增强对比度、二值化等操作"""
    # 提高图像分辨率
    image = image.convert("RGB")
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)  # 更新为LANCZOS

    # 图像增强：增强对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    # 将图像转换为灰度图像并进行二值化
    image = image.convert('L')  # 转为灰度图
    image = image.point(lambda x: 0 if x < 143 else 255, '1')  # 二值化处理（阈值设置为143）

    return image

def convert_pdf_to_text(pdf_path, output_path):
    """将 PDF 转换为文本并保存为 txt 或 markdown 格式"""
    # 使用 pdf2image 将每一页转换为图片，并设定更高的dpi值以提高图像质量
    images = convert_from_path(pdf_path, dpi=300)  # 增加分辨率

    # 将所有图片中的文本提取出来
    full_text = ""
    for page_number, image in enumerate(images):
        # 图像预处理
        processed_image = preprocess_image(image)

        # 使用 OCR 提取中文文本，指定 lang='chi_sim' 来使用中文简体语言包
        text = pytesseract.image_to_string(processed_image, lang='chi_sim')
        full_text += f"\n\nPage {page_number + 1}:\n{text}"

    # 根据需求选择输出文件格式
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(full_text)

    print(f"已处理 {pdf_path}，输出文本文件：{output_path}")

def main():
    # 检查输入目录是否存在
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：{INPUT_DIR} 不是一个目录")
        return

    # 如果指定了输出目录且不存在，则创建
    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取输入目录中所有 PDF 文件的列表
    pdf_files = glob.glob(os.path.join(INPUT_DIR, '*.pdf'))

    # 逐一处理每个 PDF 文件
    for pdf in pdf_files:
        base, ext = os.path.splitext(pdf)
        if OUTPUT_DIR:
            # 如果指定了输出目录，使用原文件名保存
            output_txt = os.path.join(OUTPUT_DIR, os.path.basename(base) + '.txt')
            output_md = os.path.join(OUTPUT_DIR, os.path.basename(base) + '.md')
        else:
            # 否则，在原目录保存，文件名后加 "_ocr" 后缀
            output_txt = base + '_ocr.txt'
            output_md = base + '_ocr.md'

        try:
            # 对 PDF 文件应用 OCRmyPDF 处理，转换为文本文件
            convert_pdf_to_text(pdf, output_txt)
            print(f"已处理 {pdf} 到 {output_txt}")
            # 可选：也可以保存为 Markdown 格式，或者直接使用相同的文本输出
            convert_pdf_to_text(pdf, output_md)
            print(f"已处理 {pdf} 到 {output_md}")
        except Exception as e:
            print(f"处理 {pdf} 时出错：{e}")

if __name__ == '__main__':
    main()
