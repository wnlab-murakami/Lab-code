import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_input('input.png', width=2, height=32, name='input'),  # 入力データ
    to_BiLSTM('bilstm', 32, 128, offset="(2,0,0)", to="(input-east)", caption="BiLSTM", width=2, height=32, depth=32),
    to_Attention('attention', 32, offset="(2,0,0)", to="(bilstm-east)", caption="Attention", width=2, height=32, depth=16),
    to_Dense('dense', 32, offset="(2,0,0)", to="(attention-east)", caption="Dense", width=1, height=32, depth=8),
    to_output('output.png', offset="(2,0,0)", to="(dense-east)", name='output'),  # 出力
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()