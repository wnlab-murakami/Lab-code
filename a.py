import matplotlib.pyplot as plt
import numpy as np

def create_question_block_art():

    # 1. カラーパレットの定義
    palette = {
        'O': (255, 223, 0),    # 黄色（本体）
        ' ': (255, 255, 255),  # 背景（白）
        'B': (142, 114, 0),    # 茶色
        'G': (192, 192, 192),  # 灰色
        'W': (255, 255, 255),   # 白
        'k': (0, 0, 0),        # 黒（リベット・影）
    }

    # 2. 設計図の定義
    question_block_design = [
        "kkkkkkkkkkkkkkkkk",
        "kWOOOOOOOOOOOOOWk",
        "kOBOOOOOOOOOOOBOk",
        "kOOOOWWWWWWWOOOOk",
        "kOOOWWWWWWWWWOOOk",
        "kOOOWWWGGGWWWOOOk",
        "kOOOWWWOOOWWWOOOk",
        "kOOOGGGOOOWWWOOOk",
        "kOOOOOOOWWWWGOOOk",
        "kOOOOOOWWWGGOOOOk",
        "kOOOOOOGGGOOOOOOk",
        "kOOOOOOOOOOOOOOOk",
        "kOOOOOOWWWOOOOOOk",
        "kOOOOOOWWWOOOOOOk",
        "kOBOOOOGGGOOOOBOk",
        "kWOOOOOOOOOOOOOWk",
        "kkkkkkkkkkkkkkkkk",


    ]

    # 3. 設計図を画像データに変換
    height = len(question_block_design)
    width = len(question_block_design[0])
    image_data = np.zeros((height, width, 3), dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            character = question_block_design[h][w]
            image_data[h, w] = palette[character]

    # 4. Matplotlibで画像を描画
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_data, interpolation='nearest')
    ax.axis('off')
    fig.canvas.manager.set_window_title('Question Block Art')
    plt.show()

if __name__ == '__main__':
    create_question_block_art()