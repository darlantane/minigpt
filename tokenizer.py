class CharTokenizer:

    def __init__(self, text: str):
        # Liste triée des caractères uniques du corpus
        self.chars = sorted(set(text))

        # Taille du vocabulaire
        self.vocab_size = len(self.chars)

        # Character -> Integer
        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        # Integer -> Character
        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    def encode(self, text: str) -> list[int]:
        """
        Convertit une chaîne de caractères en liste d'entiers.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """
        Convertit une liste d'entiers en chaîne de caractères.
        """
        return "".join(self.itos[i] for i in ids)