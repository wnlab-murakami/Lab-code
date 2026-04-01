import numpy as np

# ファイルからデータを読み込む
data = np.loadtxt('/home/dl-box/Documents/murakami/py/learning_data/noise_itf1/input/real/real_input_1_1.txt')

# FFTを実行
fft_result = np.fft.fft(data)

# 結果を表示
print(fft_result)

# 結果をファイルに保存
np.savetxt('/home/dl-box/Documents/murakami/py/python/fft_result.txt', fft_result, delimiter=',')