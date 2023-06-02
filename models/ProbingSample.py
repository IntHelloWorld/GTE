class ProbingSample:
    def __init__(self, vec, height, leaves, not_leaves, size) -> None:
        self.vec = vec
        self.height = height
        self.leaves = leaves
        self.not_leaves = not_leaves
        self.size = size

    def get_json(self):
        return {
            "vec": self.vec,
            "height": self.height,
            "leaves": self.leaves,
            "not_leaves": self.not_leaves,
            "size": self.size,
        }
