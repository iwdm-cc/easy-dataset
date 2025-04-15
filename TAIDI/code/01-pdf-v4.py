import os
import glob
from pdfminer.high_level import extract_text

# 定义输入和输出目录路径
INPUT_DIR = r"C:\Users\admin\PycharmProjects\easy-dataset\TAIDI\data"
OUTPUT_DIR = r"C:\Users\admin\PycharmProjects\easy-dataset\TAIDI\data4"


def convert_pdf_to_text_with_pdfminer(pdf_path, output_path):
    """使用 pdfminer 提取 PDF 中的文本"""
    text = extract_text(pdf_path)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

    print(f"已处理 {pdf_path}，输出文本文件：{output_path}")


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
            convert_pdf_to_text_with_pdfminer(pdf, output_txt)
            print(f"已处理 {pdf} 到 {output_txt}")
            convert_pdf_to_text_with_pdfminer(pdf, output_md)
            print(f"已处理 {pdf} 到 {output_md}")
        except Exception as e:
            print(f"处理 {pdf} 时出错：{e}")


if __name__ == '__main__':
    main()
