import pickle

def save(obj, name):
    with open('{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get(name):
    try:
        with open('{}.pickle'.format(name), 'rb') as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return {}
