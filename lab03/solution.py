from pathlib import Path
from collections import defaultdict


def read_data(dirname: str = './dataset') -> defaultdict:
    dataset = defaultdict(list)

    def read_file(file: Path) -> list:
        with file.open() as f:
            res = [int(word) for line in f for word in line.replace('Subject:', '').split()]

        return res

    def is_spam_msg(file: Path) -> int:
        if 'spmsg' in file.name:
            return 0
        return 1

    for part in Path(dirname).iterdir():
        if not part.is_dir():
            continue

        for msg in part.iterdir():
            if not msg.is_file():
                continue

            dataset[part.name].append((read_file(msg), is_spam_msg(msg)))

    return dataset


class NaiveBayes:
    def fit(self):
        pass

    def predict(self):
        pass


def k_fold_cross_validation():
    pass


if __name__ == '__main__':
    data = read_data()
    print()
    k_fold_cross_validation()
