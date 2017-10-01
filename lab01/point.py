class Point:
    def __init__(self, x: float, y: float, category: int):
        self.x = x
        self.y = y
        self.category = category

    def __str__(self) -> str:
        return 'x={}; y={}, category={}'.format(self.x, self.y, self.category)
