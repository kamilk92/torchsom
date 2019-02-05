class MapDimension:
    def __init__(self, rows: int, cols: int, features: int):
        self.__cols = cols
        self.__features = features
        self.__rows = rows

    @property
    def cols(self) -> int:
        return self.__cols

    @property
    def features(self):
        return self.__features

    @property
    def rows(self) -> int:
        return self.__rows
