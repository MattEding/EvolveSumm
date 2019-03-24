import pathlib


def extract(path):
    with open(path) as fp:
        lines = iter(fp.readlines()[:-1])
        line = next(lines)
        while line != 'Abstract:\n':
            line = next(lines)

        abstract = []
        while line != 'Introduction:\n':
            line = next(lines)
            abstract.append(line)

        original = [line for line in lines]
        return ''.join(abstract).strip(), ''.join(original).strip()



# cwd = pathlib.Path.cwd()
# path = cwd / 'data' / 'duc'
#
# abstracts = []
# originals = []
# for article in path.iterdir():
#     try:
#         abst, orig = extract(article)
#         if len(abst.split()) < 10 or len(orig.split()) < 10:
#             continue
#     except StopIteration:
#         continue
#     else:
#         abstracts.append(abst)
#         originals.append(orig)
