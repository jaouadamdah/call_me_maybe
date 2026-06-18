import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, model_validator, ConfigDict, Field

from .NextTokenSelector import NextTokenSelector
from .state_generator import StateGenerator


class Generator(BaseModel):
    """
    Manages the token-by-token generation process with constrained decoding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    states: StateGenerator
    get_next_token: NextTokenSelector
    raw_vocab: dict[str, int]
    max_token: int | None = None
    vocab: dict[int, str] = Field(default_factory=dict, init=False)

    @model_validator(mode="after")
    def build_vocab(self) -> "Generator":
        """
        Builds a reverse vocabulary mapping from token IDs to tokens.
        """

        self.vocab = {id: token for token, id in self.raw_vocab.items()}
        return self

    def call(self, prompt: str) -> str:
        """Executes the constrained generation loop for a given prompt.

        Args:
            prompt (str): The initial string prompt.

        Returns:
            str: The generated text.
        """

        state_id = 0
        tokens = 0
        ids: list[int] = self.model.encode(prompt)[0].tolist()
        generated: int = len(ids)

        print(self.encode(prompt), ids, sep='\n')

        while not self.states.is_end(state_id) and (
            not self.max_token or tokens < self.max_token
        ):
            logits = np.array(self.model.get_logits_from_input_ids(ids))
            token_id, state_id = self.get_next_token.run(
                state_id,
                logits,
            )
            ids.append(token_id)
            print(self.vocab[token_id], end="", flush=True)
            if self.max_token:
                tokens += 1
        print()
        if (
            not self.states.is_end(state_id)
            and self.max_token
            and tokens >= self.max_token
        ):
            print("[Warning] Max tokens reached!")

        res: str = self.model.decode(ids[generated:])
        return res

    def encode(self, text: str) -> list[int]:
        """Encodes a string into token IDs.

        Args:
            text (str): The input string.

        Returns:
            list[int]: A list of token IDs.
        """
        ids = []
        start = 0
        length = len(text)
        end = length

        while start != end:
            if self.raw_vocab.get(text[start:end]):
                print("found:", text[start:end].__repr__())
                ids.append(self.raw_vocab[text[start:end]])
                start = end
                end = length
            else:
                end -= 1

        return ids
