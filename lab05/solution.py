def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_data(filename='arcene_train'):
    data_filename = filename + '.data'
    labels_filename = filename + '.labels'

    labels = []
    features = []

    with open(labels_filename) as f:
        for line in f:
            labels.extend([int(num) for num in line.split(' ')])

    with open(data_filename) as f:
        for line in f:
            features.extend([int(num) for num in line.strip().split(' ')])

    features = list(chunks([num for num in features], len(features) // len(labels)))

    return features, labels


if __name__ == '__main__':
    features, labels = read_data()
    print()
