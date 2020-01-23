import pandas as pd
import os
import sys

#
# 각 개별 코드를 AST로 변경
# bcb_funcs_all.tsv : 코드 조각이 기술된 파일
#

class CodeToAST:
    def __init__(self, root):
        self.root = root
        self.sources = None
        self.size = None
        self.pairs = None

    def parse_source(self, output_file, option):
        path = self.root + output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            import javalang

            def parse_program(func):
                tokens = javalang.tokenizer.tokenize(func)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                return tree

            source = pd.read_csv(self.root + 'bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
            source.columns = ['id', 'code']
            source['code'] = source['code'].apply(parse_program)
            source.to_pickle(path)
        self.sources = source
        return source

    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl', option='existing')


code2ast = CodeToAST('data/')
code2ast.run()
