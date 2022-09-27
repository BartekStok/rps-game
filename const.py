from enum import Enum


class TransformationTypes(Enum):
    normal = "normal"
    hands = "hands"
    fingers = "fingers"

    def list(self):
        pass

    @classmethod
    def set(cls):
        return set(map(lambda t: t.value, cls))


if __name__ == "__main__":
    print(TransformationTypes.set())
    print(TransformationTypes.normal.value)
    print(TransformationTypes)
