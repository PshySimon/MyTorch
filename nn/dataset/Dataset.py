class Dataset:
    def __init__(self, *args, **kwargs):
        pass
 
    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class {}"
                                  .format('__getitem__', self.__class__.__name__))
 
    def __len__(self):
        raise NotImplementedError("'{}' not implement in class {}"
                                  .format('__len__', self.__class__.__name__))
 